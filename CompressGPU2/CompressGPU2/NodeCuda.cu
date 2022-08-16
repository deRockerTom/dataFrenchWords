#include "NodeCuda.cuh"

NodeCuda::NodeCuda(char c) {
	this->c = c;
	this->isFinal = false;
	this->indexNextStart = 0;
	this->indexNextEnd = 0;
}

void NodeCuda::init(char c, int indexNextStart, int indexNextEnd) {
	this->c = c;
	this->isFinal = false;
	this->indexNextStart = indexNextStart;
	this->indexNextEnd = indexNextEnd;
}

int NodeCuda::getIndexNextStart() {
	return this->indexNextStart;
}

int NodeCuda::getIndexNextEnd() {
	return this->indexNextEnd;
}

bool NodeCuda::isFinalNode() {
	return this->isFinal;
}

void NodeCuda::setFinalNode(bool isFinal) {
	this->isFinal = isFinal;
}

void NodeCuda::setIndexNextStart(int indexNextStart) {
	this->indexNextStart = indexNextStart;
}

void NodeCuda::setIndexNextEnd(int indexNextEnd) {
	this->indexNextEnd = indexNextEnd;
}

char NodeCuda::getChar() {
	return this->c;
}