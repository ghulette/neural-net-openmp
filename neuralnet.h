/*
 *  neuralnet.h
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/9/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

class NeuralNet {
public:
	NeuralNet(int n_inputs, int n_hidden, int n_outputs);
	~NeuralNet();
	void train(double *input, double *desired_output, double eta);
	double *run(double *input);
	double mse(double *dv);
private:
	int num_inputs;
	int num_hidden;
	int num_outputs;	
	double **hidden_weights;
	double *hidden_weights_buffer;
	double *hidden_weight_deltas;
	double *hidden_values;
	double **output_weights;
	double *output_weights_buffer;
	double *output_weight_deltas;
	double *output_values;
};

#endif
