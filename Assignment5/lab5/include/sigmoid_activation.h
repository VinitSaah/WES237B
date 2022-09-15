#ifndef __SIGMOID_ACTIVATION_H__
#define __SIGMOID_ACTIVATION_H__

#include <iostream>

#include "nn_layer.h"

using namespace std;

class SigmoidActivation : public NNLayer {
private:
	Matrix output;

	Matrix input;
	Matrix errorToLayerAbove;

public:
	SigmoidActivation ( string name );
	~SigmoidActivation ();
	
	Matrix& forward(Matrix& input);
	Matrix& backprop(Matrix& errorToLayerAbove, float learning_rate = 0.01);

	string getName() {return this->name;};
};

#endif
