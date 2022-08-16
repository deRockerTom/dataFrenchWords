
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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

struct dictionnaryEntry
{
	char* word;
	char* value;
};
struct dictionnary {
	dictionnaryEntry* entries;
	int size;
};

unordered_map<string, int> getDictWithFile(string path) {
    unordered_map<string, int> dict;
    ifstream file(path);
    string line;
    string word = line;
    string key = "";
    string value = "";
    bool isKey = true;
    if (!file.is_open()) {
        cout << " Failed to open" << endl;
    }
    else {
        cout << "Opened OK" << endl;
    }
    while (getline(file, line)) {
        for (int i = 0; i < line.length(); i++) {
            string testline = line.substr(i, 3);
            if (line.substr(i, 3)._Equal("áž€")) {
                isKey = false;
                i += 2;
            }
            else if (line.substr(i, 3)._Equal("áž")) {
                isKey = true;
                dict[key] = stoi(value);
                key = "";
                value = "";
                i += 2;
            }
            else if (isKey) {
                key += line[i];
            }
            else {
                value += line[i];
            }
        }
        key += '\n';
    }
    cout << "Dict size: " << dict.size() << endl;
    return dict;
}

char* getStringFromTxt(string path) {
    string str;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        str += line + '\n';
    }
	int n = str.length();
	char *cstr = new char[n + 1];
	strcpy(cstr, str.c_str());
	return cstr;
}

string convertToBinary(int n)
{
    string binary = "";
    while (n > 0)
    {
        binary = to_string(n % 2) + binary;
        n = n / 2;
    }
    return binary;
}

char* convertToNineBits(string bits)
{
    size_t n = bits.length();
    if (n < 9) {
        for (int i = 0; i < 9 - n; i++) {
            bits = '0' + bits;
        }
        char* cstr = new char[bits.length() + 1];
		strcpy(cstr, bits.c_str());
		return cstr;
    }
    else if (n == 9) {
		char* cstr = new char[bits.length() + 1];
		strcpy(cstr, bits.c_str());
		return cstr;
    }
    else {
        cout << "Attention : un des encodages est trop grand (> 9)" << endl;
        cout << bits << endl;
        return new char[bits.length() + 1];
    }
}

dictionnary getRealDict(unordered_map<string, int> dic, int offset) {
	const int size = dic.size();
	dictionnary *realDict;
	realDict = new dictionnary[256];
    realDict->entries = new dictionnaryEntry[size + 1];
	realDict->size = size;
	int i = 0;
	for (auto it = dic.begin(); it != dic.end(); ++it) {
		int nWord = it->first.length();
		char *word = new char[nWord + 1];
		strcpy(word, it->first.c_str());
		realDict->entries[i].word = word;
		realDict->entries[i].value = convertToNineBits(convertToBinary(i + offset));
        i++;
	}
    realDict->entries[i].word = '\0';
    realDict->entries[i].value = '\0';
	return *realDict;
}

char* searchMap(char* keys[], char* values[], char* key) {
	int i = 0;
	while (keys[i] != '\0') {
		if (strcmp(keys[i], key) == 0) {
			return values[i];
		}
		i++;
	}
	return NULL;
}

class Node {
    char c;
    Node* next;
    char* nextChar;
    bool* isFinal;
    int n;
public:
    Node(char c = '\0') {
        this->c = c;
        this->isFinal = (bool*)malloc(sizeof(bool));
        this->isFinal[0] = false;
		this->next = (Node*)malloc(sizeof(Node) * 255);
        this->nextChar = (char*)malloc(sizeof(char) * 255);
        this->n = 0;
    }
    void addEdge(char c, Node* &to) {
        this->next[this->n] = *to;
        this->nextChar[this->n] = c;
		this->n++;
    }
    Node* getEdge(char c) {
        for (int i = 0; i < n; i++) {
			if (nextChar[i] == c) {
				return &next[i];
			}
		}
		return NULL;
    }
    bool isFinalNode() {
        return this->isFinal[0];
    }
    void setFinalNode(bool setFinal) {
        this->isFinal[0] = setFinal;
    }
    char getChar() {
        return c;
    }
    Node* getNext() {
        return next;
    }
	char* getNextChar() {
		return nextChar;
	}
	int getN() {
		return n;
	}
};

class SuffixTree {
    Node* root;
public:
    SuffixTree(char* keys[], char* values[], int n) {
        this->root = new Node();
        for (int i = 0; i < n; i++) {
            Node* curr = this->root;
            for (int j = 0; j < strlen(keys[i]); j++) {
				char c = keys[i][j];
                Node* next = curr->getEdge(c);
                if (next == NULL) {
					Node* newNode = new Node(c);
                    curr->addEdge(c, newNode);
					curr = newNode;
				}
				else {
					curr = next;
                }
            }
            curr->setFinalNode(true);
            int testint = 0;
        }
    }
	

    void formatTree(Node* node) {
		Node* curr = node;
        if (curr != NULL) {
            if (curr->getN() != 0) {
                curr->getNextChar()[curr->getN()] = '\0';
            }
            for (int i = 0; i < curr->getN(); i++) {
				formatTree(&curr->getNext()[i]);
			}
        }
    }
	
    char* longestPrefix(char* s) {
        Node* current = root;
		char* prefix = (char*)malloc(sizeof(char) * 255);
		char* temp = (char*)malloc(sizeof(char) * 255);
		int i = 0;
        int j = 0;
        while (i < strlen(s) && current->getNextChar()[0] != NULL) {
            char test = s[i];
            current = current->getEdge(s[i]);
            if (current == NULL) {
				break;
			}
            else {
				temp[i] = s[i];
                if (current->isFinalNode()) {
                    strcpy(prefix, temp);
                    j = i + 1;
                }
				i++;
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
};

char* compress(char* toCompress, SuffixTree tree, dictionnary realDict) {
	char* compressed = (char*)malloc(sizeof(char) * strlen(toCompress) * 9);
	char* prefix = (char*)malloc(sizeof(char) * 255);
	char* temp = (char*)malloc(sizeof(char) * 10);
	int i = 0;
	int j = 0;
	while (i < strlen(toCompress)) {
		prefix = tree.longestPrefix(toCompress + i);
		int n = strlen(prefix);
        if (n == 0) {
		    temp = convertToNineBits(convertToBinary(toCompress[i]));
            for (int k = 0; k < 9; k++) {
				compressed[j] = temp[k];
				j++;
			}
            i++;
        }
        else {
			temp = searchMap(&realDict.entries->word, &realDict.entries->value, prefix);
            for (int k = 0; k < 9; k++) {
                compressed[j] = temp[k];
                j++;
            }
			i += n;
        }
	}
    compressed[j] = '\0';
	free(prefix);
    free(temp);
	return compressed;
}

int toDecimal(char* binary) {
	int decimal = 0;
	int i = 0;
	while (binary[i] != '\0') {
		decimal = decimal * 2 + binary[i] - '0';
		i++;
	}
	return decimal;
}


char* decompress (char* toDeCompress, char* keys[], char* values[], int size)
{
	char* decompressed = (char*)malloc(sizeof(char)*size);
	int i = 0;
    int j = 0;
	char* temp = (char*)malloc(sizeof(char)*9);
    char* temp2;
    while (j < (size / 9) * 9) {
		memcpy(temp, toDeCompress + i, 9);
		temp[9] = '\0';
		int key = toDecimal(temp);
        if (key > 255) {
			temp2 = searchMap(keys, values, temp);
			int lenTemp2 = strlen(temp2);
            memcpy(decompressed + i, temp2, lenTemp2);
			i += lenTemp2;
        }
        else {
			decompressed[i] = (char) key;
			i++;
		}
        j += 9;
    }
	decompressed[i] = '\0';
	return decompressed;
}

char* addZerosToGetBinary(char* s, int n) {
    char* binary = (char*)malloc(n + 7);
	memcpy(binary, s, n);
    if (n % 8 != 0) {
        for (int i = 0; i < 8 - n % 8; i++) {
            binary[i + n] = '0';
        }
    }
    return binary;
}

void writeToBinaryFile(char* s, string path, int size) {
    ofstream file;
    cout << "Writing to file : " << path << size << "bytes" << endl;
    file.open(path, ios::binary);
    file.write(s, size);
    file.close();
}

char* encrypt(char* s) {
	int len = strlen(s);
    int testlenchar = sizeof(char);
    char* encrypted = (char *) malloc (sizeof(char) * len/8);
    int offset;
    int temp;
    char* testTemp = (char *) malloc (sizeof(char) * 9);
    for (int i = 0; i < (len / 8); i++) {
        offset = 128;
        temp = 0;
		memcpy(testTemp, s + i * 8, 8);
        for (int j = 0; j < 8; j++) {
            temp += ((int)s[i * 8 + j] - (int)'0') * offset;
            offset /= 2;
        }
        char testchar = (char)temp;
        encrypted[i] = (char)temp;
    }
    free(testTemp);
    return encrypted;
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


    // My part

    /*int test = 255;
    string testString = convertToNineBits(convertToBinary(test));
    cout << testString << endl;
    cout << "Hello World!\n";
    unordered_map<string, string> dic;
    dic["ab"] = "01";
    dic["abc"] = "011";
    dic["coucou"] = "0111";
    dic["coucoua"] = "0111";
    dic["aba"] = "01111";
    dic["aaa"] = "011111";
    SuffixTree s = SuffixTree(dic);
    cout << s.longestPrefix("coucouabc") << endl;*/
    string path = "C:\\Users\\tomde\\Desktop\\Ecole\\n7\\Dossier_Globalink_stage_etranger\\confirmation\\Documents_recherche\\Codes\\C++\\dataFrenchWords-main";
    dictionnary dic;
    dic = getRealDict(getDictWithFile(path + "\\dicByHugoMax50"), 255);
    SuffixTree s = SuffixTree(&dic.entries->word, &dic.entries->value, dic.size);
    char* toCompress = getStringFromTxt(path + "\\anderson_contes_tome1_source.txt");
    auto startSeq = chrono::high_resolution_clock::now();
    char* compressed = compress(toCompress, s, dic);
    char* binary = addZerosToGetBinary(compressed, strlen(compressed));
    writeToBinaryFile(encrypt(binary), path + "\\anderson_contes_tome1_source.bin", strlen(compressed) / 8);
    auto endSeq = chrono::high_resolution_clock::now();
    auto durationSeq = chrono::duration_cast<chrono::milliseconds>(endSeq - startSeq);
    cout << "Sequential time : " << durationSeq.count() << "ms" << endl;
	char* decompressed = decompress(binary, &dic.entries->word, &dic.entries->value, strlen(compressed));
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