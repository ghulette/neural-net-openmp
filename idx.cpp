/*
 *  idx.cpp
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/8/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#include "idx.h"
#include <netinet/in.h>

inline int read_int(std::ifstream &ifs) {
	int buffer;
	ifs.read((char *)&buffer, sizeof(int));
	return ntohl(buffer);
}

IdxData::IdxData(std::string file) {
	using namespace std;
	
	cout << "Reading IDX file: " << file << endl;
	
	ifstream ifs(file.c_str(), ios::in | ios::binary);
	if(ifs.is_open()) {
		int magic = read_int(ifs);
		int num_dims = (magic - 0x800) - 1;
		n_records = read_int(ifs);
		record_len = 1;
		for(int i=0; i < num_dims; i++) {
			record_len *= read_int(ifs);
		}
		//cout << "  found " << n_records << " records of size " << record_len << endl;
		data = new unsigned char[n_records * record_len];
		for(int i=0; i < n_records; i++) {
			ifs.read((char *)&data[i * record_len], record_len);
			if(ifs.bad()) {
				cerr << "Error reading data" << endl;
			}
		}
		ifs.close();
	}
	else cerr << "Could not open file " << file << endl;
}

IdxData::~IdxData() {
	delete [] data;
}

int IdxData::num_records() {
	return n_records;
}

int IdxData::record_size() {
	return record_len;
}

unsigned char *IdxData::record(int record_num) {
	return &data[record_num * record_size()];
}
