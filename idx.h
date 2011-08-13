/*
 *  idx.h
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/8/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef IDX_H_
#define IDX_H_

#include <iostream>
#include <fstream>
#include <string>

class IdxData {
public:
	IdxData(std::string file);
	~IdxData();
	int num_records();
	int record_size();
	unsigned char *record(int record_num);
	
private:
	unsigned char *data;
	int n_records;
	int record_len;
};

#endif