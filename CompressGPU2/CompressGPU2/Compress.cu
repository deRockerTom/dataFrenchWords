#include "Compress.cuh"

struct dictionnaryEntry;
struct dictionnary;

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
            if (line.substr(i, 3)._Equal("ក")) {
                isKey = false;
                i += 2;
            }
            else if (line.substr(i, 3)._Equal("ខ")) {
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
    char* cstr = new char[n + 1];
    strcpy(cstr, str.c_str());
    return cstr;
}

__host__ __device__ char* convertToBinary(int n)
{
    char* binary = (char*)malloc(sizeof(char) * 10);
    int i = 0;
    while (n > 0)
    {
        binary[i] = (n % 2) + '0';
        i++;
        n = n / 2;
    }
	binary[i] = '\0';
    return binary;
}

__host__ __device__ char* convertToNineBits(char* bits)
{
    size_t n = mystrlen(bits);
    if (n < 9) {
		for (int i = n - 1; i >= 0; i--)
            bits[i] = bits[9 - n + i];
		for (int i = 0; i < 9 - n; i++)
			bits[i] = '0';
		bits[9] = '\0';
        return bits;
    }
    else if (n == 9) {
        return bits;
    }
    else {
        printf("Attention : un des encodages est trop grand (> 9)\n");
        for (int i = 0; i < mystrlen(bits); i++)
			printf("%c", bits[i]);
		printf("\n");
        return bits;
    }
}

char* compress(char* toCompress, SuffixTree tree, dictionnary realDict) {
    const int NB_THREADS = 1;
    char* compressed;
	char* resultCompressed = (char*)malloc(sizeof(char) * mystrlen(toCompress) * 9);
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMallocManaged((void**)&compressed, sizeof(char) * mystrlen(toCompress) * 9 + 1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return NULL;
    }
	
    char* temp;
	cudaStatus = cudaMallocManaged((void**)&temp, sizeof(char) * mystrlen(toCompress) * 10 * NB_THREADS);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	
    char* prefix;
	cudaStatus = cudaMallocManaged((void**)&prefix, sizeof(char) * mystrlen(toCompress) * 50 * NB_THREADS);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return NULL;
    }
	
    char* toCompressCuda;
	cudaStatus = cudaMallocManaged((void**)&toCompressCuda, sizeof(char) * strlen(toCompress) + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	cudaMemcpy(toCompressCuda, toCompress, sizeof(char) * strlen(toCompress) + 1, cudaMemcpyHostToDevice);
	
    dictionnary dic;
	cudaStatus = cudaMallocManaged((void**)&dic, sizeof(dictionnary)/*sizeof(int) + sizeof(dictionnaryEntry) * (realDict.size + 1)*/);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	
	cudaStatus = cudaMallocManaged((void**)(&dic.entries), sizeof(dictionnaryEntry) * (realDict.size + 1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	
	/*cudaStatus = cudaMallocManaged((void**)(&dic.size), sizeof(int) * (realDict.size + 1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}*/
	
    for (int i = 0; i < realDict.size; i++) {
        dic.entries[i]= realDict.entries[i];
    }
	
    for (int i = 0; i < realDict.size; i++) {
        dic.entries[i].word = realDict.entries[i].word;
        dic.entries[i].value = realDict.entries[i].value;
    }
    dic.size = realDict.size;
	
	SuffixTree treeCuda;
	cudaStatus = cudaMallocManaged((void**)&treeCuda, sizeof(SuffixTree));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	treeCuda = tree;


    NodeCuda* nodeCuda = tree.convertTree();
    NodeCudaArray nodeTree = NodeCudaArray(nodeCuda, tree.getN());
	
	NodeCudaArray nodeCudaTree;
	cudaStatus = cudaMallocManaged((void**)&nodeCudaTree, sizeof(NodeCudaArray));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	for (int i = 0; i < tree.getN(); i++) {
		nodeCudaTree.array[i] = nodeTree.array[i];
	}
	/*cudaStatus = cudaMalloc((void**)&nodeCudaTree, (sizeof(char) + sizeof(int) * 2 + sizeof(bool)) * tree.getN());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	cudaStatus = cudaMemcpy(&nodeCudaTree, &nodeTree, (sizeof(char) + sizeof(int) * 2 + sizeof(bool)) * tree.getN(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}*/
	
	/*cudaStatus = cudaMemcpy(&nodeCudaCuda, nodeCuda, sizeof(NodeCuda) * tree.getN(), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return NULL;
    }*/	

	/*cudaStatus = cudaMemcpy(&dic, (void**)&realDict, sizeof(dictionnary), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//return NULL;
	}
	cudaStatus = cudaMemcpy(&dic.entries, (void**)&realDict.entries, sizeof(dictionnaryEntry) * realDict.size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//return NULL;
	}
	cudaStatus = cudaMemcpy(&dic.size, (void**)&realDict.size, sizeof(int) * realDict.size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return NULL;
	}*/
	
    compressWithCuda<<<1,1>>>(toCompressCuda, nodeCudaTree, dic, compressed, temp, prefix);
	
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "compress launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compressWithCuda!\n", cudaStatus);
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
	
	cudaMemcpy(resultCompressed, compressed, sizeof(char) * mystrlen(toCompress) * 9, cudaMemcpyDeviceToHost);
	
    return resultCompressed;
}

__global__ void compressWithCuda(char* toCompress, NodeCudaArray tree, dictionnary realDict, char* compressed, char* temp, char* prefix)
{
    int i = 0;
    int j = 0;
	int toCompressLen = mystrlen(toCompress);
    printf("%d\n", realDict.size);
    printf("%d test len toCompress\n", toCompressLen);
    //printf("%d test first tree len\n", (tree.array[0]).getIndexNextEnd() - (tree.array[0]).getIndexNextStart() + 1);
	//printf("%d test second tree len\n", (tree.array[1]).getIndexNextEnd()- (tree.array[1]).getIndexNextStart() + 1);
    while (i < toCompressLen) {
        printf("test boucle\n");
		prefix = tree.longestPrefix(toCompress + i);
        printf("test before\n");
        int n = mystrlen(prefix);
        printf("%d\n", n);
        if (n == 0) {
            printf("coucou\n");
            temp = convertToNineBits(convertToBinary(toCompress[i]));
            for (int k = 0; k < 9; k++) {
                compressed[j] = temp[k];
                j++;
            }
            i++;
        }
        else {
            printf("coucou 2\n");
            printf(prefix);
            temp = realDict.searchMap(prefix);
            printf("coucou 3\n");
            for (int k = 0; k < 9; k++) {
                compressed[j] = temp[k];
                j++;
            }
            i += n;
        }
    }
    printf("%d\n", j);
    printf("%d\n", i);
    compressed[j] = '\0';
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


char* decompress(char* toDeCompress, dictionnary dic, int size)
{
    char* decompressed = (char*)malloc(sizeof(char) * size);
    int i = 0;
    int j = 0;
    char* temp = (char*)malloc(sizeof(char) * 9);
    char* temp2;
    while (j < (size / 9) * 9) {
        memcpy(temp, toDeCompress + i, 9);
        temp[9] = '\0';
        int key = toDecimal(temp);
        if (key > 255) {
            temp2 = dic.searchMap(temp);
            int lenTemp2 = strlen(temp2);
            memcpy(decompressed + i, temp2, lenTemp2);
            i += lenTemp2;
        }
        else {
            decompressed[i] = (char)key;
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
    char* encrypted = (char*)malloc(sizeof(char) * len / 8);
    int offset;
    int temp;
    char* testTemp = (char*)malloc(sizeof(char) * 9);
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

__host__ __device__ int mystrlen(char* s) {
	int i = 0;
	while (s[i] != '\0') {
		i++;
	}
	return i;
}

__host__ __device__ void mystrcpy(char* &dest, char* src) {
	int i = 0;
	while (src[i] != '\0') {
		dest[i] = src[i];
		i++;
	}
	dest[i] = '\0';
}

__host__ __device__ int mystrcmp(const char* str_a, const char* str_b, unsigned len) {
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
	printf("debut mystrcmp\n");
	printf("%c", str_a[0]);
    while ((i < len) && (match == 0) && !done) {
        if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
        else if (str_a[i] != str_b[i]) {
            match = i + 1;
            if (((int)str_a[i] - (int)str_b[i]) < 0) match = 0 - (i + 1);
        }
        i++;
    }
    return match;
}
