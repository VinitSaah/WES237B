#ifndef __MSECOST_H__
#define __MSECOST_H__

#include "matrix.h"

class MSECost {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};

#endif
