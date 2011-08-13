/*
 *  neuralnet.cpp
 *  neuralnet
 *
 *  Created by Geoff Hulette on 2/9/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#include "neuralnet.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cycle.h"

#ifdef _OPENMP
#include <omp.h>
#endif

inline double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

inline double rand_weight() {
  return 0.5 - (double(rand()) / RAND_MAX);
}

NeuralNet::NeuralNet(int n_inputs, int n_hidden, int n_outputs) {
  num_inputs = n_inputs;
  num_hidden = n_hidden;
  num_outputs = n_outputs;
  
  hidden_values = new double[num_hidden];
  hidden_weight_deltas = new double[num_hidden];
  hidden_weights_buffer = new double[num_hidden * num_inputs];
  hidden_weights = new double*[num_hidden];
  
  #pragma omp parallel for
  for(int i=0; i < num_hidden; i++) {
    hidden_weights[i] = hidden_weights_buffer + (i * num_inputs);
    for(int j=0; j < num_inputs; j++) {
      hidden_weights[i][j] = rand_weight();
    }
  }
  
  output_values = new double[num_outputs];
  output_weight_deltas = new double[num_outputs];
  output_weights_buffer = new double[num_outputs * num_hidden];
  output_weights = new double*[num_outputs];
  
  #pragma omp parallel for
  for(int i=0; i < num_outputs; i++) {
    output_weights[i] = output_weights_buffer + (i * num_hidden);
    for(int j=0; j < num_hidden; j++) {
      output_weights[i][j] = rand_weight();
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel 
    {
#pragma omp single
    std::cout << "Using OpenMP with " << omp_get_num_threads() 
              << " threads" << std::endl;
    }
#endif
}

NeuralNet::~NeuralNet() {
  delete[] hidden_weights;
  delete[] hidden_weights_buffer;
  delete[] hidden_weight_deltas;
  delete[] hidden_values;

  delete[] output_weights;
  delete[] output_weights_buffer;
  delete[] output_weight_deltas;
  delete[] output_values;
}

double *NeuralNet::run(double *input) {
  double *iv = input;
  double *hv = hidden_values;
  double *ov = output_values;
  
  // Propagate from input to hidden layer
  #pragma omp parallel for
  for(int i=0; i < num_hidden; i++) {
    double sum = 0.0;
    for(int j=0; j < num_inputs; j++) {
      sum += iv[j] * hidden_weights[i][j];
    }
    hv[i] = sigmoid(sum);
  }
  
  // Propagate from hidden to output layer
  #pragma omp parallel for
  for(int i=0; i < num_outputs; i++) {
    double sum = 0.0;
    for(int j=0; j < num_hidden; j++) {
      sum += hv[j] * output_weights[i][j];
    }
    ov[i] = sigmoid(sum);
  }
  
  return ov;
}

void NeuralNet::train(double *input, double *desired_output, double eta) {
  run(input);
  double *iv = input;
  double *hv = hidden_values;
  double *ov = output_values;
  double *dv = desired_output;
  
  // Calculate weight deltas for the output layer
  #pragma omp parallel for
  for(int i=0; i < num_outputs; i++) {
    output_weight_deltas[i] = (dv[i] - ov[i]) * ov[i] * (1.0 - ov[i]);
  }
    
  // Calculate weight deltas for the hidden layer
  #pragma omp parallel for
  for(int i=0; i < num_hidden; i++) {
    double sum = 0.0;
    for(int j=0; j < num_outputs; j++) {
      sum += output_weight_deltas[j] * output_weights[j][i];
    }
    hidden_weight_deltas[i] = sum * hv[i] * (1.0 - hv[i]);
  }
        
  // Update output layer weights
  #pragma omp parallel for
  for(int i=0; i < num_outputs; i++) {
    for(int j=0; j < num_hidden; j++) {
      output_weights[i][j] += eta * output_weight_deltas[i] * hv[j];
    }
  }
            
  // Update hidden layer weights
  #pragma omp parallel for
  for(int i=0; i < num_hidden; i++) {
    for(int j=0; j < num_inputs; j++) {
      hidden_weights[i][j] += eta * hidden_weight_deltas[i] * iv[j];
    }
  }
}

double NeuralNet::mse(double *desired_values) {
  double sum = 0.0;
  for(int i=0; i < num_outputs; i++) {
    double diff = desired_values[i] - output_values[i];
    sum += diff * diff;
  }
  return sum;
}
