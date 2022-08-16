#pragma once
#ifndef __NODE_H__
#define __NODE_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class Node {
	char c;
	Node* next;
	char* nextChar;
	bool* isFinal;
	int n;
public:
			   __host__ Node(char c = '\0');
	__device__ __host__ void init(char c);
	__device__ __host__ void addEdge(char c, Node*& to);
	__device__ __host__ Node* getEdge(char c);
	__device__ __host__ bool isFinalNode();
	__device__ __host__ void setFinalNode(bool setFinal);
	__device__ __host__ char getChar();
	__device__ __host__ Node* getNext();
	__device__ __host__ char* getNextChar();
	__device__ __host__ int getN();
};

#endif