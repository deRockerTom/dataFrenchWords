#pragma once
#ifndef __NODECUDA_H__
#define __NODECUDA_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class NodeCuda {
	char c;
	int indexNextStart;
	int indexNextEnd;
	bool isFinal;
public:
	__host__ NodeCuda(char c = '\0');
	__device__ __host__ void init(char c, int indexNextStart, int indexNextEnd);
	__device__ __host__ int getIndexNextStart();
	__device__ __host__ int getIndexNextEnd();
	__device__ __host__ bool isFinalNode();
	__device__ __host__ void setFinalNode(bool setFinal);
	__device__ __host__ void setIndexNextStart(int indexNextStart);
	__device__ __host__ void setIndexNextEnd(int indexNextEnd);
	__device__ __host__ char getChar();
};

#endif