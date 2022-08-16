#pragma once
#ifndef __DICTIONNARY_H__
#define __DICTIONNARY_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dictionnaryEntry.cuh"
#include <iostream>
#include <unordered_map>

class dictionnary {
public:
	dictionnaryEntry *entries;
	int size;
	__host__ __device__ dictionnary();
	__host__ dictionnary(std::unordered_map<std::string, int> dic, int offset);
	__host__ __device__ char* searchMap(char* key);
};

#endif