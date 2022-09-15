#ifndef __LINEARLAYER_H__
#define __LINEARLAYER_H__

#include "nn_layer.h"

using namespace std;

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;
	
	Matrix output;
	Matrix input;
	Matrix eA; //error to layer above

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();
	void initializeWeightsHalf();

	void computeAndStoreBackpropError(Matrix& dG);
	void computeAndStoreLayerOutput(Matrix& X);
	void updateWeights(Matrix& dG, float learning_rate);
	void updateBias(Matrix& dG, float learning_rate);

public:
	LinearLayer(string name, Shape W_shape);
	~LinearLayer();

	Matrix& forward(Matrix& X);
	Matrix& backprop(Matrix& eB, float learning_rate = 0.01); //eB is error from layer below

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;

};

#endif
