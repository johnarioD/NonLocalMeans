
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fstream>

using namespace std;

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
float expf(float x);
#endif

__global__ void cuFirstPart(float* image, float* Wi, int size, int batch_size, int stride, float sigma, int pixel);
__global__ void cuSum(float* Wi, int size);
__global__ void cuBlockSum(float* Wi, float* Zi, int size, int length);
__global__ void cuThirdPart(float* image, float* Wi, float* div, int size);

void NLM(float* image, float* clearImage, int size, int stride, int batch_size, float sigma);
void readcsv(float* A, char *name, int size);
void writecsv(float* A, char* name, int size, int stride);

int main()
{
    float sigma = 1;
    int batch_size = 3;
    int stride = 256;
    int size = stride*stride;

    float *image;
    image = (float*)malloc(size * sizeof(float));
    float* clearImage;
    clearImage = (float*)malloc(size * sizeof(float));

    readcsv(image, "C:/Users/Giannis/Desktop/PDS/Ex3/testingNoise.csv", stride);

    NLM(image, clearImage, size, stride, batch_size, sigma);
        
    writecsv(clearImage, "C:/Users/Giannis/Desktop/PDS/Ex3/Dumb.csv", size, stride);

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
    dim3 sumGrid(grid.x / 2, grid.y);

    float* Wi, * Zi;
    Wi = (float*)malloc(grid.y * size * sizeof(float));
    Zi = (float*)malloc(grid.y * sizeof(float));

    float* d_image = NULL;
    float* d_Wi = NULL;
    float* d_Zi = NULL;

    err = cudaMalloc((void**)&d_image, size * sizeof(float));
    if (err != cudaSuccess) printf("Error at d_image malloc\n%s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_Wi, grid.y * size * sizeof(float));
    if (err != cudaSuccess) printf("Error at d_Wi malloc\n%s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_Zi, grid.y * sizeof(float));
    if (err != cudaSuccess) printf("Error at d_Zi malloc\n%s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_image, image, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Error at image memcpy\n%s\n", cudaGetErrorString(err));

    //-------------------------------------------------------------------------------------------------
    // 
    //     ITERATE PIXELS:
    //
    auto beg = std::chrono::high_resolution_clock::now();
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
        if (err != cudaSuccess) printf("Error at cuFirstPart\n%s\n", cudaGetErrorString(err));

        err = cudaMemcpy(Wi, d_Wi, grid.y * size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("Error at Wi memcpy\n%s\n", cudaGetErrorString(err));
        //-------------------------------------------------------------------------------------------------
        //
        //      SUM_1:
        //
        cuSum << <sumGrid, TpB >> > (d_Wi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error at cuSum\n%s\n", cudaGetErrorString(err));

        cuBlockSum << <sumGrid.y, sumGrid.x / 2 >> > (d_Wi, d_Zi, size, 2 * TpB);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error at cuBlockSum\n%s\n", cudaGetErrorString(err));

        err = cudaMemcpy(d_Wi, Wi, grid.y * size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("Error at d_Wi memcpy\n%s\n", cudaGetErrorString(err));

        err = cudaMemcpy(Zi, d_Zi, grid.y * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("Error at d_Zi memcpy\n%s\n", cudaGetErrorString(err));
        //-------------------------------------------------------------------------------------------------
        //
        //     CALCULATE:
        //      Wi = Wi./Zi;
        //      Wi = Wi*image;
        //    
        cuThirdPart << <grid, TpB >> > (d_image, d_Wi, d_Zi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error at cuThirdPart\n%s\n", cudaGetErrorString(err));
        //-------------------------------------------------------------------------------------------------
        //
        //      SUM_2:
        //    
        cuSum << <sumGrid, TpB >> > (d_Wi, size);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error at cuSum 2\n%s\n", cudaGetErrorString(err));

        cuBlockSum << <sumGrid.y, sumGrid.x / 2 >> > (d_Wi, d_Zi, size, 2 * TpB);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error at cuBlockSum 2\n%s\n", cudaGetErrorString(err));

        err = cudaMemcpy(Zi, d_Zi, grid.y * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("Error at d_Zi memcpy\n%s\n", cudaGetErrorString(err));
        //-------------------------------------------------------------------------------------------------
        //
        //      COPY RESULTS:
        //
        memcpy(clearImage + pixel, Zi, grid.y * sizeof(float));
        //-------------------------------------------------------------------------------------------------
    }
    auto fin = std::chrono::high_resolution_clock::now();
    //
    //                              --IMAGE HAS BEEN DENOISED--
    //
    //-------------------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------------------
    //
    //      FREE MEMORY:
    //
    err = cudaFree(d_image);
    if (err != cudaSuccess) printf("Error at d_image free\n%s\n", cudaGetErrorString(err));
    err = cudaFree(d_Wi);
    if (err != cudaSuccess) printf("Error at d_Wi free\n%s\n", cudaGetErrorString(err));
    err = cudaFree(d_Zi);
    if (err != cudaSuccess) printf("Error at d_Zi free\n%s\n", cudaGetErrorString(err));
    free(Zi);
    free(Wi);

    std::chrono::duration<float> duration = fin - beg;
    printf("Duration : %f\n", duration.count());   
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

                if ((rowI + X >= 0) && (colI + Y >= 0) &&
                    (rowI + X < stride) && (colI + Y < stride)) {
                    Ni = image[(rowI + X) * stride + colI + Y];

                }
                if ((rowJ + X >= 0) && (colJ + Y >= 0) &&
                    (rowJ + X < stride) && (colJ + Y < stride)) {
                    Nj = image[(rowJ + X) * stride + colJ + Y];

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

__global__ void cuThirdPart(float* image, float* Wi, float *Zi, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int rowStart = blockIdx.y * size;

    if (i < size) {
        Wi[rowStart + i] = Wi[rowStart + i] * image[i] / Zi[blockIdx.y];
    }
}
