
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Compress.cuh"
#include "SuffixTree.cuh"
#include "Node.cuh"
#include "dictionnaryEntry.cuh"
#include "dictionnary.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <chrono>

using namespace std;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }




	// MY PART

    string path = "C:\\Users\\tomde\\Desktop\\Ecole\\n7\\Dossier_Globalink_stage_etranger\\confirmation\\Documents_recherche\\Codes\\C++\\dataFrenchWords-main";
    unordered_map<string, int> firstDic = getDictWithFile(path + "\\dicByHugoMax50");
    dictionnary dic = dictionnary(firstDic, 255);
    SuffixTree s = SuffixTree(&dic.entries->word, &dic.entries->value, dic.size);
    char* toCompress = getStringFromTxt(path + "\\anderson_contes_tome1_source.txt");
    auto startSeq = chrono::high_resolution_clock::now();
    char* compressed = compress(toCompress, s, dic);
    cudaStatus = cudaDeviceReset();
    char* binary = addZerosToGetBinary(compressed, strlen(compressed));
    writeToBinaryFile(encrypt(binary), path + "\\anderson_contes_tome1_source.bin", strlen(compressed) / 8);
    auto endSeq = chrono::high_resolution_clock::now();
    auto durationSeq = chrono::duration_cast<chrono::milliseconds>(endSeq - startSeq);
    cout << "Sequential time : " << durationSeq.count() << "ms" << endl;
    char* decompressed = decompress(binary, dic, strlen(compressed));
    writeToBinaryFile(decompressed, path + "\\anderson_contes_tome1_source_decompressed.txt", strlen(decompressed));
    //auto startPar = chrono::high_resolution_clock::now();
    /*string compressedPar = compressParralel(toCompress, s, dic);
    addZerossToGetBinary(compressedPar);
    writeToBinaryFile(encryptParallel(compressedPar), path + "\\anderson_contes_tome1_source_par.bin");
    auto endPar = chrono::high_resolution_clock::now();
    auto durationPar = chrono::duration_cast<chrono::milliseconds>(endPar - startPar);
    cout << "Parallel time : " << durationPar.count() << "ms" << endl;
    cout << "Speedup : " << (double)durationSeq.count() / durationPar.count() << endl;
    /*string toDeCompress = getStringFromTxt(path + "\\anderson_contes_tome1_source.bin");
    string decompressed = decompress(toDeCompress, dic);*/
    cout << "end" << endl;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
