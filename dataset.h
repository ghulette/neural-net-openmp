/*
 *  dataset.h
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/9/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>

class DataSet {
public:
	enum {OUTPUT_ASCII, OUTPUT_PACKED, OUTPUT_FULL};
	
	DataSet(std::string idx_image_file, std::string idx_label_file, int n_categories);
	~DataSet();
	double *image_vector(int index);
	double *label_vector(int index);
	int label(int index);
	int image_vector_length();
	int label_vector_length();
	int length();
	void print_image(int n, int width, int height);
	void print_label(int n);
	int label_for_vector(double *vec);
private:
	int num_entries;
	int image_data_length;
	int label_data_length;
	double **image_data;
	double **label_data;
	int *labels;
};

#endif
