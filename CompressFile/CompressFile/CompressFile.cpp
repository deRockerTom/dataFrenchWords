// CompressFile.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <algorithm>

using namespace std;

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


string getStringFromTxt(string path) {
	string str;
	ifstream file(path);
	string line;
	while (getline(file, line)) {
		str += line + '\n';
	}
	return str;
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

string convertToNineBits(string bits)
{
	size_t n = bits.length();
	if (n < 9) {
		for (int i = 0; i < 9 - n; i++) {
			bits = '0' + bits;
		}
		return bits;
	}
	else if (n == 9) {
		return bits;
	}
	else {
		cout << "Attention : un des encodages est trop grand (> 9)"	<< endl;
		cout << bits << endl;
		return "";
	}
}

unordered_map<string, string> getRealDict(unordered_map<string, int> dic, int offset) {
	unordered_map<string, string> realDict;
	int i = 0;
	for (auto& pair : dic) {
		realDict[pair.first] = convertToNineBits(convertToBinary(i++ + offset));
	}
	return realDict;
}

class Node {
		char c;
		unordered_map<char, Node*> out;
		bool isFinal;
	public:
		Node(char c = '\0') {
		this->c = c;
		isFinal = false;
	}
		void addEdge(char c, Node* to) {
		out[c] = to;
	}
		Node* getEdge(char c) {
		return out[c];
	}
		bool isFinalNode() {
		return isFinal;
	}
		void setFinalNode(bool isFinal) {
		this->isFinal = isFinal;
	}
		char getChar() {
		return c;
	}
		unordered_map<char, Node*>& getOut() {
			unordered_map<char, Node*>& outref = out;
			return outref;
	}
};

class SuffixTree {
		Node *root;
	public:
		SuffixTree(unordered_map<string, string> dic) {
			root = new Node();
			for (auto it = dic.begin(); it != dic.end(); it++) {
				string word = it->first;
				string code = it->second;
				Node* current = root;
				// loop over the word char
				for (int i = 0; i < word.length(); i++) {
					char c = word[i];
					if (current->getOut().size() == 0) {
						Node* n = new Node(c);
						current->addEdge(c, n);
						current = n;
					}
					else if (current->getOut().find(c) == current->getOut().end()) {
						Node* n = new Node(c);
						current->addEdge(c, n);
						current = n;
					}
					else {
						current = (current->getEdge(c));
					}
				}
				current->setFinalNode(true);
				
			}
		}
		string longestPrefix(string s) {
			Node* current = root;
			string word = "";
			string temp = "";
			bool end = false;
			int i = 0;
			size_t length = s.length();
			while (i < length && !end) {
				char c = s[i];
				if (current->getOut().find(c) == current->getOut().end()) {
					end = true;
				}
				else {
					temp += c;
					current = (current->getEdge(c));
					if (current->isFinalNode()) {
						word = temp;
					}
					i++;
				}
			}
			return word;
		}

		string longestPrefixParallel(string s) {
			Node* current = root;
			string word = "";
			string temp = "";
			bool end = false;
			int i = 0;
			size_t length = s.length();
			while (i < length && !end) {
				char c = s[i];
				if (current->getOut().find(c) == current->getOut().end()) {
					end = true;
				}
				else {
					temp += c;
					current = (current->getEdge(c));
					if (current->isFinalNode()) {
						word = temp;
					}
					i++;
				}
			}
			return word;
		}
};

string compress(string toCompress, SuffixTree tree, unordered_map<string, string> dic) {
	string compressed = "";
	size_t i = 0;
	size_t length = toCompress.length();
	while (i < length) {
		string temp = tree.longestPrefix(toCompress.substr(i));
		size_t n = temp.length();
		if (n == 0) {
			string testString = convertToNineBits(convertToBinary((int)toCompress[i]));
			compressed += convertToNineBits(convertToBinary((int)toCompress[i]));
			i++;
		}
		else {
			string testString = dic[temp];
			compressed += dic[temp];
			i += n;
		}
	}
	return compressed;
}

string compressParralel(string toCompress, SuffixTree tree, unordered_map<string, string> dic, int nbThreads = thread::hardware_concurrency()) {
	string compressed = "";
	size_t length = toCompress.length();
#pragma omp parallel for num_threads(nbThreads) private(temp, substring, sublength, n, i) reduction(+= : compressed) //shared(toCompress, tree, dic, length)
	for (int t = 0; t < nbThreads; t++) {
		size_t i = 0;
		size_t sublength = min(length / nbThreads + 1, length - t * length / nbThreads);
		string substring = toCompress.substr(t * length/nbThreads, sublength);
		cout << "Thread " << t << endl;
		while (i < sublength) {
			string temp = tree.longestPrefix(substring.substr(i));
			size_t n = temp.length();
			if (n == 0) {
				compressed += convertToNineBits(convertToBinary((int)substring[i]));
				i++;
			}
			else {
				compressed += dic[temp];
				i += n;
			}
		}
		cout << "Thread " << t << " finished" << endl;
	}
	return compressed;
}

template<typename K, typename V>
unordered_map<V, K> inverse_map(unordered_map<K, V>& map)
{
	unordered_map<V, K> inv;
	std::for_each(map.begin(), map.end(),
		[&inv](const pair<K, V>& p) {
			inv.insert(make_pair(p.second, p.first));
		});
	return inv;
}


string decompress(string toDeCompress, unordered_map<string, string> dic) {
	string decompressed = "";
	unordered_map<string, string> inverted_dic = inverse_map(dic);
	int i = 0;
	size_t length = toDeCompress.length();
	while (i < length) {
		string temp = toDeCompress.substr(i, 9);
		int n = stoi(temp, nullptr, 2);
		if (n > 255) {
			decompressed += inverted_dic[temp];
			i += 9;
		}
		else {
			decompressed += (char)n;
			i += 9;
		}
	}
	return decompressed;
}

void addZerossToGetBinary(string &s) {
	size_t n = s.length();
	if (n % 8 != 0) {
		for (int i = 0; i < 8 - n % 8; i++) {
			s += '0';
		}
	}
}

void writeToBinaryFile(string s, string path) {
	ofstream file;
	cout << "Writing to file : " << path << s.length() << "bits" << endl;
	file.open(path, ios::binary);
	file.write(s.c_str(), s.length());
	file.close();
}

void writeToFile(string s, string path) {
	ofstream file;
	cout << "Writing to file : " << path << s.length() << "bits" << endl;
	const char* test = s.c_str();
	file.open(path);
	file.write(s.c_str(), s.length());
	file.close();
}
string encrypt(string s) {
	string encrypted = "";
	int offset;
	int temp;
	for (int i = 0; i < (s.length() / 8); i++) {
		offset = 128;
		temp = 0;
		string testTemp = s.substr(i * 8, 8);
		for (int j = 0; j < 8; j++) {
			temp += ((int)s[i * 8 + j] - (int)'0') * offset;
			offset /= 2;
		}
		encrypted += (char)temp;
	}
	return encrypted;
}

string encryptParallel(string s) {
	string encrypted = "";
	int offset;
	int temp;
#pragma omp parallel for private(temp) reduction(+= : encrypted)
	for (int i = 0; i < (s.length() / 8); i++) {
		offset = 128;
		temp = 0;
		for (int j = 0; j < 8; j++) {
			temp += ((int)s[i * 8 + j] - (int)'0') * offset;
			offset /= 2;
		}
		encrypted += (char)temp;
	}
	return encrypted;
}

int main()
{
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
	unordered_map<string, string> dic;
	dic = getRealDict(getDictWithFile(path + "\\dicByHugoMax50"), 255);
	SuffixTree s = SuffixTree(dic);
	string toCompress = getStringFromTxt(path + "\\anderson_contes_tome1_source.txt");
	auto startSeq = chrono::high_resolution_clock::now();
	string compressed = compress(toCompress, s, dic);
	addZerossToGetBinary(compressed);
	writeToBinaryFile(encrypt(compressed), path + "\\anderson_contes_tome1_source.bin");
	auto endSeq = chrono::high_resolution_clock::now();
	auto durationSeq = chrono::duration_cast<chrono::milliseconds>(endSeq - startSeq);
	cout << "Sequential time : " << durationSeq.count() << "ms" << endl;
	auto startPar = chrono::high_resolution_clock::now();
	string compressedPar = compressParralel(toCompress, s, dic);
	addZerossToGetBinary(compressedPar);
	writeToBinaryFile(encryptParallel(compressedPar), path + "\\anderson_contes_tome1_source_par.bin");
	auto endPar = chrono::high_resolution_clock::now();
	auto durationPar = chrono::duration_cast<chrono::milliseconds>(endPar - startPar);
	cout << "Parallel time : " << durationPar.count() << "ms" << endl;
	cout << "Speedup : " << (double)durationSeq.count() / durationPar.count() << endl;
	//string toDeCompress = getStringFromTxt(path + "\\anderson_contes_tome1_source.bin");
	string decompressed = decompress(compressed, dic);
	writeToFile(decompressed, path + "\\anderson_contes_tome1_source_decompressed.txt");
	cout << "end" << endl;
	
	return 0;
	
	

	
	
}



// Exécuter le programme : Ctrl+F5 ou menu Déboguer > Exécuter sans débogage
// Déboguer le programme : F5 ou menu Déboguer > Démarrer le débogage

// Astuces pour bien démarrer : 
//   1. Utilisez la fenêtre Explorateur de solutions pour ajouter des fichiers et les gérer.
//   2. Utilisez la fenêtre Team Explorer pour vous connecter au contrôle de code source.
//   3. Utilisez la fenêtre Sortie pour voir la sortie de la génération et d'autres messages.
//   4. Utilisez la fenêtre Liste d'erreurs pour voir les erreurs.
//   5. Accédez à Projet > Ajouter un nouvel élément pour créer des fichiers de code, ou à Projet > Ajouter un élément existant pour ajouter des fichiers de code existants au projet.
//   6. Pour rouvrir ce projet plus tard, accédez à Fichier > Ouvrir > Projet et sélectionnez le fichier .sln.
