#include "SuffixTree.cuh"
#include "Node.cuh"
#include "Compress.cuh"
#include "NodeCuda.cuh"
#include <queue>

SuffixTree::SuffixTree() {
	this->root = new Node();
    this->root->init('\0');
    this->n = 1;
}

SuffixTree::SuffixTree(char* keys[], char* values[], int n) {
    this->root = new Node();
    /*if (err != cudaSuccess) {
        printf("Error allocating memory for root\n");
    }*/
    this->root->init('\0');
    this->n = 1;
    for (int i = 0; i < n; i++) {
        Node* curr = this->root;
        for (int j = 0; j < mystrlen(keys[i]); j++) {
            char c = keys[i][j];
            Node* next = curr->getEdge(c);
            if (next == NULL) {
                Node* newNode;
                newNode = new Node();
                /*if (err != cudaSuccess) {
                    printf("Error allocating memory for newNode");
                }*/
                newNode->init(c);
                curr->addEdge(c, newNode);
                curr = newNode;
                this->n++;
            }
            else {
                curr = next;
            }
        }
        curr->setFinalNode(true);
        int testint = 0;
    }
}

char* SuffixTree::longestPrefix(char* s) {
    Node* current = root;
    char* prefix = (char*)malloc(sizeof(char) * 255);
    char* temp = (char*)malloc(sizeof(char) * 255);
    // Length of temp
    int i = 0;
    // Length of the longest Prefix + 1
    int j = 0;
    while (i < mystrlen(s) && current->getNextChar()[0] != NULL) {
        char test = s[i];
        current = current->getEdge(s[i]);
        // if we are at the end of the tree
        if (current == NULL) {
            break;
        }
        else {
            temp[i] = s[i];
            // If we found a good word, replace temp with it and update j
            if (current->isFinalNode()) {
                mystrcpy(prefix, temp);
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

void SuffixTree::formatTree(Node* node) {
    Node* curr = node;
    if (curr != NULL) {
        if (curr->getN() != 0) {

        }
        for (int i = 0; i < curr->getN(); i++) {
            formatTree(&curr->getNext()[i]);
        }
    }
}

NodeCuda* SuffixTree::convertTree() {
    NodeCuda* result = (NodeCuda*)malloc(sizeof(NodeCuda) * this->n);
    Node* curr = root;
    // Index of current node
    int i = 0;
    // Index of next node of current node
    int j = 1;
	std::queue<Node*> q;
	q.push(curr);
	while (!q.empty()) {
		Node* curr = q.front();
		q.pop();
        if (curr->getN() != 0) {
            result[i].init(curr->getChar(), j, j + curr->getN() - 1);
        }
        else {
			result[i].init(curr->getChar(), 0, 0);
		}
		result[i].setFinalNode(curr->isFinalNode());
        i++;
		for (int k = 0; k < curr->getN(); k++) {
			q.push(&curr->getNext()[k]);
            j++;
		}
	}
	return result;
}

int SuffixTree::getN() {
	return this->n;
}