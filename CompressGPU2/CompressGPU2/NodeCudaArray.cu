#include "NodeCudaArray.cuh"
#include "Compress.cuh"

NodeCudaArray::NodeCudaArray(NodeCuda* array, int size) {
	this->array = new NodeCuda[size];
	for (int i = 0; i < size; i++) {
		this->array[i] = array[i];
	}
}

__host__ __device__ int NodeCudaArray::getEdge(int indexStart, int indexEnd, char c) {
    int i = indexStart;
    while (i < indexEnd) {
        if (this->array[i].getChar() == c) {
            return i;
        }
        i++;
    }
    return -1;
}

__host__ __device__ char* NodeCudaArray::longestPrefix(char* s) {
    NodeCuda current = this->array[0];
    char* prefix = (char*)malloc(sizeof(char) * 255);
    char* temp = (char*)malloc(sizeof(char) * 255);
    // Length of temp
    int i = 0;
    // Length of the longest Prefix + 1
    int j = 0;
    while (i < mystrlen(s) && current.getIndexNextStart() != 0) {
        int index = getEdge(current.getIndexNextStart(), current.getIndexNextEnd(), s[i]);
        if (index != -1) {
            current = this->array[index];
            temp[i] = s[i];
            if (current.isFinalNode()) {
                mystrcpy(prefix, temp);
                j = i + 1;
            }
            i++;
        }
        else {
            break;
        }
    }
    char* returnPrefix;
    returnPrefix = (char*)malloc(sizeof(char) * (j + 1));
    memcpy(returnPrefix, prefix, j);
    returnPrefix[j] = '\0';
    free(prefix);
    free(temp);
    return returnPrefix;
}