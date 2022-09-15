#include <iostream>
#include <time.h>

#include "dataset.h"
#include "matrix.h"
#include "neural_network.h"

#include "linear_layer.h"
#include "sigmoid_activation.h"

#define PRINT_WEIGHTS 0

using namespace std;

float computeAccuracy(const Matrix& predictions, const Matrix& targets, const Matrix& batches);
float getPrediction(float pred);
void printPredictions(const Matrix& predictions, const Matrix& targets, const Matrix& bathces);
int printWeights(LinearLayer ll);
int printBias(LinearLayer ll);

int main(int argc, const char* argv[]) {

	srand ( time(NULL));

	int batch_size = 4;
	int num_batches = 3;

	if(argc >= 3)
	{
		batch_size = atoi(argv[1]);	
		num_batches = atoi(argv[2]);	
	}

	Dataset dataset(batch_size, num_batches); //batch_size , number of batches
	MSECost mse_cost;

	NeuralNetwork nn = NeuralNetwork(0.01f);
    //TODO: build your network structure
	LinearLayer ll1 = LinearLayer("linear_1", Shape(2, 2));
	nn.addLayer(&ll1);
	nn.addLayer(new SigmoidActivation("sigmoid_1"));
	LinearLayer ll2 = LinearLayer("linear_2", Shape(2, 1));
	nn.addLayer(&ll2);
	nn.addLayer(new SigmoidActivation("sigmoid_2"));

	Matrix Y;
	int final_epoch = 0;
	int printed_lines = 0;
	for (int epoch = 0; epoch < 5000; epoch++)
	{	
		//Clear all the previously printed lines
		for (int j = 0; j < printed_lines; j++)
		{
			printf("\33[2K\033[A");
		}
		printed_lines = 0;

		float cost = 0.0f;

		for (int batch = 0; batch < dataset.getNumOfBatches()-1; batch++)
		{
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += mse_cost.cost(Y, dataset.getTargets().at(batch));
		}

		cost = cost/(dataset.getNumOfBatches()-1);

#if PRINT_WEIGHTS
        //Below is an example of how to print weight updates for a layer
//		printed_lines += printWeights(ll1);
#endif
		printf("Cost : %f\n", cost);
		printed_lines++;
		
		if (cost < .001)
		{
			break;
		}

		final_epoch = epoch;
	}

	printf("Total number of epochs : %i\n", final_epoch);

	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches()-1));
	Y.copyDeviceToHost();
	float accuracy = computeAccuracy(Y, dataset.getTargets().at(dataset.getNumOfBatches()-1), dataset.getBatches().at(dataset.getNumOfBatches()-1));

	cout << "Accuracy: " << accuracy << endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets, const Matrix& batches)
{
	int m = predictions.shape.cols;
	int correct_predictions = 0;

	//Uncomment to print the predictions during accuracy
	printPredictions(predictions, targets, batches);

	for (int i = 0; i < m; i++)
	{
		float prediction = getPrediction(predictions.data_host.at<float>(0,i));

		if (prediction == targets.data_host.at<float>(0,i))
			correct_predictions++;
	}

	return static_cast<float>(correct_predictions) / m;
}

float getPrediction(float pred)
{
	float prediction = 0;
	prediction = pred > .5 ? 1 : 0;		
	return prediction;
}

void printPredictions(const Matrix& predictions, const Matrix& targets, const Matrix& batches)
{
	int m = predictions.shape.cols;

	for (int i = 0; i < m; i++)
	{
		float prediction = getPrediction(predictions.data_host.at<float>(0,i));
		printf("Data : [%f, %f] / Pred (pred) - Real : %f (%f) - %f\n",  batches.data_host.at<float>(0, i), batches.data_host.at<float>(1, i), predictions.data_host.at<float>(0,i), prediction, targets.data_host.at<float>(0,i));
	}
	printf("\n");
}

int printWeights(LinearLayer ll)
{
	int lines = 0;
	Matrix W = ll.getWeightsMatrix();
	W.copyDeviceToHost();
	int rows = W.data_host.rows;
	int cols = W.data_host.cols;
	printf("Weights : ");
	for (int i = 0; i < rows; i++)
	{
		printf("[");
		for (int j = 0; j < cols; j++)
		{
			printf(" %f ", W.data_host.at<float>(i,j));
		}
		printf("]\n          ");
		lines++;
	}
	printf("\n");
	lines++;
	return lines;
}

int printBias(LinearLayer ll)
{
	int lines = 0;
	Matrix b = ll.getBiasVector();
	b.copyDeviceToHost();
	int rows = b.data_host.rows;
	int cols = b.data_host.cols;
	printf("Bias (%i, %i): ", rows, cols);
	for (int i = 0; i < rows; i++)
	{
		printf("[");
		for (int j = 0; j < cols; j++)
		{
			printf(" %f ", b.data_host.at<float>(j,i));
		}
		printf("]\n          ");
		lines++;
	}
	printf("\n");
	lines++;
	return lines;
}
