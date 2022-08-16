#pragma once
#ifndef __NODECUDAARRAY_H__
#define __NODECUDAARRAY_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "NodeCuda.cuh"

class NodeCudaArray {
public:
	NodeCuda* array;
	NodeCudaArray(NodeCuda* array = NULL, int size = 0);
    __host__ __device__ int getEdge(int indexStart, int indexEnd, char c);

    __host__ __device__ char* longestPrefix(char* s);
};

#endif