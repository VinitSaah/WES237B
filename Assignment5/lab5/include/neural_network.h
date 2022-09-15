#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <vector>
#include "nn_layer.h"
#include "mse_cost.h"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	MSECost mse_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer* layer);
	std::vector<NNLayer*> getLayers() const;
};

#endif
