#include "dictionnaryEntry.cuh"

dictionnaryEntry::dictionnaryEntry() {
	this->word = (char*)malloc(sizeof(char) * 51);
	this->word[0] = '\0';
	this->value = (char*)malloc(sizeof(char) * 10);
	this->value[0] = '\0';
}

dictionnaryEntry::dictionnaryEntry(char* word, char* value) {
	this->word = (char*)malloc(sizeof(char) * 51);
	strcpy(this->word, word);
	this->value = (char*)malloc(sizeof(char) * 10);
	strcpy(this->value, value);
}