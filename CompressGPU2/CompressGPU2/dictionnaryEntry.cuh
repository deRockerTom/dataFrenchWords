#pragma once
#ifndef __DICTIONNARYENTRY_H__
#define __DICTIONNARYENTRY_H__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

class dictionnaryEntry {
public:
	char* word;
	char* value;
	__host__ dictionnaryEntry();
	__host__ dictionnaryEntry(char* word, char* value);
};

#endif