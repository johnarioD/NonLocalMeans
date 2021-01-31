#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>

using namespace std;

__global__ void cuFirstPart(float* image, float* Wi, int size, int batch_size, int stride, float sigma, int pixel);
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
            out << A[row * stride + col] << ',';
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

    int TpB = 1024;
    int blocksPerPixel = (size - 1) / TpB + 1;
    int concurrentPixels = 65536 / blocksPerPixel;
    if (size < concurrentPixels) {
        concurrentPixels = size;
    }
    dim3 grid(blocksPerPixel, concurrentPixels);


    //-------------------------------------------------------------------------------------------------
    //
    //     MALLOCS:
    //
    cudaError_t err = cudaSuccess;
    struct timespec ts_start, ts_end;
    dim3 sumGrid(grid.x / 2, grid.y);

    float* Wi, * Zi;
    Wi = (float*)malloc(grid.y * size * sizeof(float));
    Zi = (float*)malloc(grid.y * sizeof(float));

    float* d_image = NULL;
    float* d_Wi = NULL;
    float* d_Zi = NULL;

    err = cudaMalloc((void**)&d_image, size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_image malloc\n" << cudaGetErrorString(err) << "\n";
    err = cudaMalloc((void**)&d_Wi, grid.y * size * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_Wi malloc\n" << cudaGetErrorString(err) << "\n";
    err = cudaMalloc((void**)&d_Zi, grid.y * sizeof(float));
    if (err != cudaSuccess) cout << "Error at d_Zi malloc\n" << cudaGetErrorString(err) << "\n";

    err = cudaMemcpy(d_image, image, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cout << "Error at d_image memcpy\n" << cudaGetErrorString(err) << "\n";

    //-------------------------------------------------------------------------------------------------
    //
    //     ITERATE PIXELS:
    //
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int pixel = 0; pixel < size; pixel += grid.y) {
        printf("Pixels %d-%d\n", pixel, pixel + grid.y - 1);

        //-------------------------------------------------------------------------------------------------
       //
       //      CALCULATE:
       //       Wi = norm(Ni - Nj);
       //       Wi = Wi./sigma^2;
       //       Wi = exp(Wi);
       //
        cuFirstPart << <grid, TpB >> > (d_image, d_Wi, size, batch_size, stride, sigma, pixel);
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

        err = cudaMemcpy(Zi, d_Zi, grid.y * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) cout << "Error at Zi memcpy\n" << cudaGetErrorString(err) << "\n";
        //-------------------------------------------------------------------------------------------------
        //
        //      COPY RESULTS:
        //
        memcpy(clearImage + pixel, Zi, grid.y * sizeof(float));
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
    err = cudaFree(d_Wi);
    if (err != cudaSuccess) cout << "Error at d_Wi free\n" << cudaGetErrorString(err) << "\n";
    err = cudaFree(d_Zi);
    if (err != cudaSuccess) cout << "Error at d_Zi free\n" << cudaGetErrorString(err) << "\n";
    free(Zi);
    free(Wi);

    cout << "Time:" << ts_end.tv_sec - ts_start.tv_sec << "." << abs(ts_end.tv_nsec - ts_start.tv_nsec)) << "\n";
}

/*--------------------------------------------------------------------------------------------------------------------------
*
*       CUDA KERNELS:
*
--------------------------------------------------------------------------------------------------------------------------*/
__global__ void cuFirstPart(float* image, float* Wi, int size, int batch_size, int stride, float sigma, int pixel) {

    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < size) {
        int rowStart = blockIdx.y * size;
        int center = blockIdx.y + pixel;

        int rowI, colI, rowJ, colJ;
        Wi[rowStart + j] = 0;

        rowI = center / stride;
        colI = center - rowI * stride;
        rowJ = j / stride;
        colJ = j - rowJ * stride;

        colI -= batch_size / 2;
        rowI -= batch_size / 2;
        colJ -= batch_size / 2;
        rowJ -= batch_size / 2;

        for (int X = 0; X < batch_size; X++) {
            for (int Y = 0; Y < batch_size; Y++) {
                float Ni = 0;
                float Nj = 0;

                if (rowI + X >= 0){
                    if (colI + Y >= 0) {
                        if (rowI + X < stride) {
                            if (colI + Y < stride) {
                                Ni = image[(rowI + X) * stride + colI + Y];
                            }
                        }
                    }
                }
                if (rowJ + X >= 0) {
                    if (colJ + Y >= 0) {
                        if (rowJ + X < stride) {
                            if (colJ + Y < stride) {
                                Nj = image[(rowJ + X) * stride + colJ + Y];
                            }
                        }
                    }
                }
                Wi[rowStart + j] += (Ni - Nj) * (Ni - Nj);
            }
        }
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
