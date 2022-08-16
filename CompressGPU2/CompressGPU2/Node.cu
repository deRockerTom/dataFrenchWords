#include "Node.cuh"

Node::Node(char c) {
    this->c = c;
	this->isFinal = (bool*)malloc(sizeof(bool));
	/*if (err != cudaSuccess) {
		printf("Error allocating memory for isFinal\n");
	}*/
    this->isFinal[0] = false;
	this->next = (Node*)malloc(sizeof(Node*)*255);
	/*if (err != cudaSuccess) {
		printf("Error allocating memory for next\n");
	}*/
	this->nextChar = (char*)malloc(sizeof(char));
	/*if (err != cudaSuccess) {
		printf("Error allocating memory for nextChar\n");
	}*/
    this->n = 0;
}

void Node::init(char c) {
	this->c = c;
	cudaError_t err = cudaMallocManaged(&isFinal, sizeof(bool));
	if (err != cudaSuccess) {
		printf("Error allocating memory for isFinal\n");
	}
	this->isFinal[0] = false;
	err = cudaMallocManaged(&next, sizeof(Node*) * 255);
	if (err != cudaSuccess) {
		printf("Error allocating memory for next\n");
	}
	err = cudaMallocManaged(&nextChar, sizeof(char));
	if (err != cudaSuccess) {
		printf("Error allocating memory for nextChar\n");
	}
}

void Node::addEdge(char c, Node*& to) {
    this->next[this->n] = *to;
    this->nextChar[this->n] = c;
    this->n++;
}

Node* Node::getEdge(char c) {
	for (int i = 0; i < this->n; i++) {
		if (this->nextChar[i] == c) {
			return &this->next[i];
		}
	}
	return NULL;
}

bool Node::isFinalNode() {
	return this->isFinal[0];
}

void Node::setFinalNode(bool b) {
	this->isFinal[0] = b;
}

char Node::getChar() {
	return this->c;
}

Node* Node::getNext() {
	return this->next;
}

char* Node::getNextChar() {
	return this->nextChar;
}

int Node::getN() {
	return this->n;
}

