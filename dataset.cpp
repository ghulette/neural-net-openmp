/*
 *  dataset.cpp
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/9/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#include "dataset.h"
#include "idx.h"

const int ASCII_LENGTH = 8;
const int PACKED_LENGTH = 4;
const int FULL_LENGTH = 10;

double *vector_for_label_ascii(int label) {
	double *vec = new double[ASCII_LENGTH];
	unsigned char c = (unsigned char)label + 48;
	for(int i=0; i < ASCII_LENGTH; i++) {
		vec[i] = (double)(c & 1);
		c = c >> 1;
	}
	return vec;
}

int label_for_vector_ascii(double *vec) {
	unsigned char label = 0;
	for(int i=0; i < ASCII_LENGTH; i++) {
		label |= int(vec[i] + 0.5) << i;
	}
	return label - 48;
}

double *vector_for_label_packed(int label) {
	double *vec = new double[PACKED_LENGTH];
	char c = (char)label;
	for(int i=0; i < PACKED_LENGTH; i++) {
		vec[i] = (double)(c & 1);
		c = c >> 1;
	}
	return vec;
}

int label_for_vector_packed(double *vec) {
	int label = 0;
	for(int i=0; i < PACKED_LENGTH; i++) {
		label |= int(vec[i] + 0.5) << i;
	}
	return label;
}

double *vector_for_label_full(int label) {
	double *vec = new double[FULL_LENGTH];
	for(int i=0; i < FULL_LENGTH; i++) {
		vec[i] = (i == label) ? 1.0 : 0.0;
	}
	return vec;
}

int label_for_vector_full(double *vec) {
	double max = 0.0;
	int maxi = 0;
	for(int i=0; i < FULL_LENGTH; i++) {
		if(vec[i] > max) {
			max = vec[i];
			maxi = i;
		}
	}
	return maxi;
}

void print_vec(double *vec, int len) {
	using namespace std;
	cout << "Vec: ";
	for(int i=0; i < len; i++) {
		cout << int(vec[i] + 0.5) << " ";
	}
	cout << endl;
}

void test() {
	using namespace std;
	cout << "Testing full encoding" << endl;
	for(int i=0; i < 10; i++) {
		cout << "Label: " << i << endl;
		double *vec = vector_for_label_full(i);
		print_vec(vec, FULL_LENGTH);
		int label = label_for_vector_full(vec);
		cout << "Output: " << label << endl;
		delete[] vec;
	}
	cout << "Testing ascii encoding" << endl;
	for(int i=0; i < 10; i++) {
		cout << "Label: " << i << endl;
		double *vec = vector_for_label_ascii(i);
		print_vec(vec, ASCII_LENGTH);
		int label = label_for_vector_ascii(vec);
		cout << "Output: " << label << endl;
		delete[] vec;
	}
	cout << "Testing packed encoding" << endl;
	for(int i=0; i < 10; i++) {
		cout << "Label: " << i << endl;
		double *vec = vector_for_label_packed(i);
		print_vec(vec, PACKED_LENGTH);
		int label = label_for_vector_packed(vec);
		cout << "Output: " << label << endl;
		delete[] vec;
	}
	
}

DataSet::DataSet(std::string idx_image_file, std::string idx_label_file, int output_enc) {
	using namespace std;
	
	IdxData *image_idx = new IdxData(idx_image_file);
	IdxData *label_idx = new IdxData(idx_label_file);
	
	if(image_idx->num_records() != label_idx->num_records()) {
		cerr << "Image and label files have differing lengths" << endl;
		exit(1);
	}
	num_entries = image_idx->num_records();
	image_data_length = image_idx->record_size();
	
	switch(output_enc) {
		case OUTPUT_ASCII:
			label_data_length = ASCII_LENGTH;
			break;
		case OUTPUT_PACKED:
			label_data_length = PACKED_LENGTH;
			break;
		case OUTPUT_FULL:
			label_data_length = FULL_LENGTH;
			break;
		default:
			cerr << "Error: Output encoding " << output_enc << " not supported." << endl;
			exit(1);
	}
	
	image_data = new double *[num_entries];
	label_data = new double *[num_entries];
	labels = new int[num_entries];
	
	for(int i=0; i < num_entries; i ++) {
		// Label digit
		labels[i] = int(label_idx->record(i)[0]);
		
		// Label output vector
		switch(output_enc) {
			case OUTPUT_ASCII:
				label_data[i] = vector_for_label_ascii(labels[i]);
				break;
			case OUTPUT_PACKED:
				label_data[i] = vector_for_label_packed(labels[i]);
				break;
			case OUTPUT_FULL:
				label_data[i] = vector_for_label_full(labels[i]);
				break;
		}
		
		// Image vector (rescaled from 0-255 to 0.0-1.0)
		image_data[i] = new double[image_idx->record_size()];
		unsigned char *idx_img = image_idx->record(i);
		for(int j=0; j < image_data_length; j++) {
			image_data[i][j] = double(idx_img[j]) / 255.0;
		}
	}
	
	delete image_idx;
	delete label_idx;
}

DataSet::~DataSet() {
	for(int i=0; i < num_entries; i++) {
		delete[] image_data[i];
		delete[] label_data[i];
	}
	delete[] image_data;
	delete[] label_data;
	delete[] labels;
}

int DataSet::length() {
	return num_entries;
}

int DataSet::image_vector_length() {
	return image_data_length;
}

int DataSet::label_vector_length() {
	return label_data_length;
}

double *DataSet::image_vector(int index) {
	return image_data[index];
}

double *DataSet::label_vector(int index) {
	return label_data[index];
}

int DataSet::label(int index) {
	return labels[index];
}

int DataSet::label_for_vector(double *vec) {
	int label = -1;
	switch(label_data_length) {
		case ASCII_LENGTH:
			label = label_for_vector_ascii(vec);
			break;
		case PACKED_LENGTH:
			label = label_for_vector_packed(vec);
			break;
		case FULL_LENGTH:
			label = label_for_vector_full(vec);
			break;
	}
	return label;
}

void DataSet::print_image(int n, int width, int height) {
	using namespace std;
	
	double *img = this->image_vector(n);
	for(int y=0; y < height; y ++) {
		for(int x=0; x < width; x ++) {
			double pixel = img[y * height + x];
			if(pixel == 0.0) {
				cout << "  ";
			} else if(pixel < 0.9) {
				cout << "..";
			} else {
				cout << "##";
			}
		}
		cout << endl;
	}
}

void DataSet::print_label(int n) {
	using namespace std;
	
	double *label = this->label_vector(n);
	cout << "[" << label[0];
	for(int i=1; i < this->label_vector_length(); i++) {
		cout << ", " << label[i];
	}
	cout << "]" << endl;
}