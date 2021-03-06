#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>

using namespace std;

__global__ void cuFilter(float* filter, int batch_size);
__global__ void cuMeans(float* image, float* means, float* filter, int size, int batch_size, int stride);
__global__ void cuFirstPart(float* means, float* Wi, int size, float sigma, int pixel);
__global__ void cuSum(float* Wi, int size);
__global__ void cuBlockSum(float* Wi, float* Zi, int size, int length);
__global__ void cuThirdPart(float* image, float* Wi, float* div, int size);

void NLM(float* image, float* clearImage, int size, int stride, int batch_size, float sigma);
void readcsv(float* A, char* name, int size);
void writecsv(float* A, char* name, int size, int stride);

int main(int argc, char** argv)
{
    char* srcName = "./testingNoise.csv";
    char* dstName = "./testingClear.csv";
    float sigma = 0.5;
    int batch_size = 3;
    int stride = 256;

    if (argc == 6) {
        srcName = argv[1];
        dstName = argv[2];
        stride = atoi(argv[3]);
        batch_size = atoi(argv[4]);
        sigma = atof(argv[5]);
    }
    else {
        cout << "Incorrect Arguments\nTry: [Executable] <source file> <destination file> <stride> <batch size> <sigma>\n";
        return 0;
    }

    int size = stride * stride;

    float* image;
    image = (float*)malloc(size * sizeof(float));
    float* clearImage;
    clearImage = (float*)malloc(size * sizeof(float));

    readcsv(image, srcName, stride);

    NLM(image, clearImage, size, stride, batch_size, sigma);

    writecsv(clearImage, dstName, size, stride);

    free(image);
    free(clearImage);
    return 0;
}

/*--------------------------------------------------------------------------------------------------------------------------
*
*       FILE HANDLING:
*
--------------------------------------------------------------------------------------------------------------------------*/
void readcsv(float* A, char* name, int stride) {
    FILE* file = fopen(name, "r");

    for (int i = 0; i < stride; i++) {
        for (int j = 0; j < stride; j++) {
            fscanf(file, "%f,", A + i * stride + j);
        }
    }
}
void writecsv(float* A, char* name, int size, int stride) {
    std::ofstream out(name);

    for (int row = 0; row < size / stride; row++) {
        for (int col = 0; col < stride; col++) {
            out << A[row * stride + col];
            if(col < stride-1) out << ',';
        }
        out << '\n';
    }
}

/*--------------------------------------------------------------------------------------------------------------------------
*
*       HOST FUNCTIONS:
*
--------------------------------------------------------------------------------------------------------------------------*/
void NLM(float* image, float* clearImage, int size, int stride, int batch_size, float sigma) {

    float* Wi;
    struct timespec ts_start, ts_end;
    cudaError_t err = cudaSuccess;
    
    int TpB = 1024;
    int blocksPerPixel = (size - 1) / TpB + 1;
    int concurrentPixels = 65536 / blocksPerPixel;
    if (size < concurrentPixels) {
        concurrentPixels = size;
    }
    dim3 grid(blocksPerPixel, concurrentPixels);    
    dim3 sumGrid(grid.x / 2, grid.y);
    //-------------------------------------------------------------------------------------------------
    //
    //     MALLOCS:
    //
    Wi = (float*)malloc(grid.y * size * sizeof(float));

    float* d_image = NULL;
    float* d_means = NULL;
    float* d_filter = NULL;
    float* d_Wi = NULL;
    float* d_Zi = NULL;

    err = cudaMalloc((void**)&d_image, size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_image malloc\n" << cudaGetErrorString(err) << "\n";
    err = cudaMalloc((void**)&d_filter, batch_size * batch_size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_filter malloc\n" << cudaGetErrorString(err) <<"\n";
    err = cudaMalloc((void**)&d_means, size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_means malloc\n" << cudaGetErrorString(err) <<"\n";
    err = cudaMalloc((void**)&d_Wi, grid.y * size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_Wi malloc\n" << cudaGetErrorString(err) << "\n";
    err = cudaMalloc((void**)&d_Zi, grid.y * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_Zi malloc\n" << cudaGetErrorString(err) << "\n";

    err = cudaMemcpy(d_image, image, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cout << "Error at d_image memcpy\n" << cudaGetErrorString(err) << "\n";
    
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    //-------------------------------------------------------------------------------------------------
    // 
    //     FIND MEANS:
    //
    cuFilter<<<batch_size,batch_size>>>(d_filter, batch_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) cout << "Error at cuFilter\n" << cudaGetErrorString(err) << "\n";
    cuMeans << <grid.x, TpB >> > (d_image, d_means, d_filter, size, batch_size, stride);
    err = cudaGetLastError();
    if (err != cudaSuccess) cout << "Error at cuMeans\n" << cudaGetErrorString(err) << "\n";
    
    //-------------------------------------------------------------------------------------------------
    //
    //     ITERATE PIXELS:
    //
    for (int pixel = 0; pixel < size; pixel += grid.y) {
        printf("Pixels %d-%d\n", pixel, pixel + grid.y - 1);

        //-------------------------------------------------------------------------------------------------
        //
        //      CALCULATE:
        //       Wi = |Ni - Nj|^2;
        //       Wi = Wi./sigma^2;
        //       Wi = exp(Wi);
        //
        cuFirstPart << <grid, TpB >> > (d_means, d_Wi, size, sigma, pixel);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at cuFirstPart\n" << cudaGetErrorString(err) << "\n";

        err = cudaMemcpy(Wi, d_Wi, grid.y * size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) cout << "Error at Wi memcpy\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
        //
        //      SUM_1:
        //
        cuSum << <sumGrid, TpB >> > (d_Wi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at cuSum\n" << cudaGetErrorString(err) << "\n";

        cuBlockSum << <sumGrid.y, sumGrid.x / 2 >> > (d_Wi, d_Zi, size, 2 * TpB);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at cuBlockSum\n" << cudaGetErrorString(err) << "\n";

        err = cudaMemcpy(d_Wi, Wi, grid.y * size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) cout << "Error at d_Wi memcpy\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
        //
        //     CALCULATE:
        //      Wi = Wi./Zi;
        //      Wi = Wi*image;
        //
        cuThirdPart << <grid, TpB >> > (d_image, d_Wi, d_Zi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at cuThirdPart\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
        //
        //      SUM_2:
        //
        cuSum << <sumGrid, TpB >> > (d_Wi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at second cuSum\n" << cudaGetErrorString(err) << "\n";

        cuBlockSum << <sumGrid.y, sumGrid.x / 2 >> > (d_Wi, d_Zi, size, 2 * TpB);
        err = cudaGetLastError();
        if (err != cudaSuccess) cout << "Error at second cuBlockSum\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
        //
        //      COPY RESULTS:
        //
        err = cudaMemcpy(clearImage + pixel, d_Zi, grid.y * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) cout << "Error at Zi memcpy\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
    }
  	clock_gettime(CLOCK_MONOTONIC, &ts_end);
    //
    //                              --IMAGE HAS BEEN DENOISED--
    //
    //-------------------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------------------
    //
    //      FREE MEMORY:
    //
    err = cudaFree(d_image);
    if (err != cudaSuccess) cout << "Error at d_image free\n" << cudaGetErrorString(err) << "\n";
    err = cudaFree(d_means);
    if (err != cudaSuccess) cout << "Error at d_means free\n" << cudaGetErrorString(err) << "\n";
    err = cudaFree(d_Wi);
    if (err != cudaSuccess) cout << "Error at d_Wi free\n" << cudaGetErrorString(err) << "\n";
    err = cudaFree(d_Zi);
    if (err != cudaSuccess) cout << "Error at d_Zi free\n" << cudaGetErrorString(err) << "\n";
    free(Wi);

    cout << "Time:" << ts_end.tv_sec - ts_start.tv_sec << "." << abs(ts_end.tv_nsec - ts_start.tv_nsec)) << "\n";
}

/*--------------------------------------------------------------------------------------------------------------------------
*
*       CUDA KERNELS:
*
--------------------------------------------------------------------------------------------------------------------------*/
__global__ void cuFilter(float* filter, int batch_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float x = (blockIdx.x - batch_size / 2) * (blockIdx.x - batch_size / 2);
    x += (threadIdx.x - batch_size / 2) * (threadIdx.x - batch_size / 2);
    filter[i] = expf(-x/100);
}

__global__ void cuMeans(float* image, float* means, float* filter, int size, int batch_size, int stride) {
    int center = blockDim.x * blockIdx.x + threadIdx.x;

    if (center < size) {
        int top = center/stride;
        int left = center - top*stride;
        top -= batch_size / 2;
        left -= batch_size / 2;

        float count = 0;
        means[center] = 0;
        for (int Y = top; Y < top + batch_size; Y++) {
            for (int X = left; X < left + batch_size; X++) {
                if ((X >= 0) && (X < stride)
                   &&(Y >= 0) && (Y < stride)) {
                    means[center] += image[Y * stride + X]*filter[(X-left)*batch_size+Y-top];
                    count++;
                }
            }
        }
        means[center] /= count;
    }
}

__global__ void cuFirstPart(float* means, float* Wi, int size, float sigma, int pixel) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < size) {
        int rowStart = blockIdx.y * size;
        int center = blockIdx.y + pixel;

        Wi[rowStart + j] = (means[center] - means[j])* (means[center] - means[j]);

        __syncthreads();
        Wi[rowStart + j] = expf(-(Wi[rowStart + j] / (sigma * sigma)));
    }
}
__global__ void cuSum(float* Wi, int size) {

    int i = threadIdx.x;
    int blockStart = 2 * blockIdx.x * blockDim.x;
    int rowStart = blockIdx.y * size;

    if (blockStart + i < size) {
        int stride;
        for (stride = blockDim.x; stride >= 1; stride /= 2) {
            if (i < stride) {
                if (blockStart + i + stride < size) {
                    Wi[rowStart + blockStart + i] = Wi[rowStart + blockStart + i] + Wi[rowStart + blockStart + i + stride];
                }
            }
            __syncthreads();
        }
    }
}

__global__ void cuBlockSum(float* Wi, float* Zi, int size, int length) {
    int i = threadIdx.x;
    int rowStart = blockIdx.x * size;

    int stride;
    for (stride = blockDim.x; stride >= 1; stride /= 2) {
        if (i < stride) {
            if (((i + stride) * length) < size) {
                Wi[rowStart + i * length] = Wi[rowStart + i * length] + Wi[rowStart + (i + stride) * length];
            }
        }
        __syncthreads();
    }
    if (i == 0) {
        Zi[blockIdx.x] = Wi[rowStart];
    }
}

__global__ void cuThirdPart(float* image, float* Wi, float* Zi, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int rowStart = blockIdx.y * size;

    if (i < size) {
        Wi[rowStart + i] = Wi[rowStart + i] * image[i] / Zi[blockIdx.y];
    }
}
