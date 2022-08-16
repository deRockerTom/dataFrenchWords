#pragma once
#ifndef __COMPRESS_H__
#define __COMPRESS_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SuffixTree.cuh"
#include "dictionnary.cuh"
#include "NodeCuda.cuh"
#include "NodeCudaArray.cuh"

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

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

unordered_map<string, int> getDictWithFile(string path);

char* getStringFromTxt(string path);

__host__ __device__ char* convertToBinary(int n);

__host__ __device__ char* convertToNineBits(char* bits);

char* compress(char* toCompress, SuffixTree tree, dictionnary realDict);

__global__ void compressWithCuda(char* toCompress, NodeCudaArray tree, dictionnary realDict, char* compressed, char* temp, char* prefix);

int toDecimal(char* binary);

char* decompress(char* toDeCompress, dictionnary dic, int size);

char* addZerosToGetBinary(char* s, int n);

void writeToBinaryFile(char* s, string path, int size);

char* encrypt(char* s);

__host__ __device__ int mystrlen(char* s);

__host__ __device__ void mystrcpy(char* &dest, char* src);

__host__ __device__ int mystrcmp(const char* str_a, const char* str_b, unsigned len = 256);

__host__ __device__ char* longestPrefix(NodeCuda* tree, char* s);

#endif