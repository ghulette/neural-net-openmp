#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "dataset.h"
#include "neuralnet.h"
#include "cycle.h"

const std::string IDX_DIR = "idx/";
const std::string TRAINING_IMAGES = "train-images.idx3-ubyte";
const std::string TRAINING_LABELS = "train-labels.idx1-ubyte";
const std::string TESTING_IMAGES = "t10k-images.idx3-ubyte";
const std::string TESTING_LABELS = "t10k-labels.idx1-ubyte";

const double ETA = 0.3;
const int NUM_HIDDEN_NODES = 300;
const int OUTPUT_ENC = DataSet::OUTPUT_FULL;

int main(int argc, char * const argv[]) {
	using namespace std;
	
	srand((unsigned)time(0));
	cout.precision(2);
	cout.setf(ios::fixed | ios::showpoint);
	
	DataSet *train = new DataSet(IDX_DIR + TRAINING_IMAGES, IDX_DIR + TRAINING_LABELS, OUTPUT_ENC);
	DataSet *test = new DataSet(IDX_DIR + TESTING_IMAGES, IDX_DIR + TESTING_LABELS, OUTPUT_ENC);
	NeuralNet *nn = new NeuralNet(train->image_vector_length(), NUM_HIDDEN_NODES, train->label_vector_length());
	
	int train_len = train->length();
	cout << "Training on " << train_len << " examples" << endl;
	ticks train_start = getticks();
	for(int i=0; i < train_len; i++) {
		nn->train(train->image_vector(i), train->label_vector(i), ETA);
	}
	ticks train_end = getticks();
	
	int test_len = test->length();
	cout << "Testing on " << test_len << " examples" << endl;
	ticks test_start = getticks();
	int correct = 0;
	for(int i=0; i < test_len; i++) {
		int guess = test->label_for_vector(nn->run(test->image_vector(i)));
		int answer = test->label(i);
		if(guess == answer) {
			correct ++;
		}
	}
	ticks test_end = getticks();
	
	cout << "Correct: " << correct << "/" << test_len << " (" << 100 * double(correct)/test_len << "%)" << endl;
	cout << "Training: " << elapsed(train_end, train_start) << endl;
	cout << "Testing: " << elapsed(test_end, test_start) << endl;
	
	delete nn;
	delete train;
	delete test;

    return 0;
}
