#pragma once
#ifndef __SUFFIXTREE_H__
#define __SUFFIXTREE_H__
#include "Node.cuh"
#include "NodeCuda.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class SuffixTree {
	Node* root;
	int n;
public:
			   __host__ SuffixTree();
			   __host__ SuffixTree(char* keys[], char* values[], int n);
	__device__ __host__ void formatTree(Node* node);
			   __host__ NodeCuda* convertTree();
	__device__ __host__ int getN();
	__device__ __host__ char* longestPrefix(char* s);
};

#endif