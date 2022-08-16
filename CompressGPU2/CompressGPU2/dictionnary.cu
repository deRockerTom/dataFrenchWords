#include "dictionnary.cuh"
#include "Compress.cuh"

dictionnary::dictionnary() {
    const int size = 255;
    this->entries = new dictionnaryEntry[255 + 1];
    this->size = 255;
    this->entries[256].word = '\0';
    this->entries[256].value = '\0';
}

dictionnary::dictionnary(std::unordered_map<std::string, int> dic, int offset) {
    const int size = dic.size();
    this->entries = new dictionnaryEntry[size + 1];
    this->size = size;
    int i = 0;
    for (auto it = dic.begin(); it != dic.end(); ++it) {
        int nWord = it->first.length();
        char* word = new char[nWord + 1];
        strcpy(word, it->first.c_str());
        this->entries[i].word = word;
        this->entries[i].value = convertToNineBits(convertToBinary(i + offset));
        i++;
    }
    this->entries[i].word = '\0';
    this->entries[i].value = '\0';
}

__host__ __device__ char* dictionnary::searchMap(char* key) {
    int i = 0;
    //dictionnaryEntry* entries = this->entries;
    while (this->entries[i].word != '\0') {
        printf("test\n");
        //printf("%c", this->entries[i].word[0]);
        if (mystrcmp(this->entries[i].word, key) == 0) {
            printf("return searchmap");
            return this->entries[i].value;
        }
        i++;
    }
    return NULL;
}