#ifndef __DATASET_H__
#define __DATASET_H__


#include "matrix.h"

#include <vector>

class Dataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:
	Dataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();
};

#endif
