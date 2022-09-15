#ifndef __NNLAYER_H__
#define __NNLAYER_H__

#include <iostream>

#include "matrix.h"

using namespace std;

class NNLayer {
protected:
	string name;

public:
	virtual ~NNLayer() = 0;
	
	virtual Matrix& forward(Matrix& X) = 0;
	virtual Matrix& backprop(Matrix& dY, float learning_rate) = 0;

	string getName() {return this->name;};
};

inline NNLayer::~NNLayer() {}

#endif
