#include "main.h"
using namespace std;
// ----------------------------------------------
int main(int argc, const char * argv[])
// ----------------------------------------------
{
    const int SAMPLES    = 100000; // Size of input data
    const int FILTER_LEN = 64; // Size of filter

    // Initialize coefficients
    float coeffs[FILTER_LEN];

    designLPF(coeffs, FILTER_LEN, 44.1, 2.0);

    ifstream ifile("input.txt");
    ifstream gfile("golden_output.txt");

    // Create inputs
    float input[SAMPLES];
    for(int i = 0; i < SAMPLES; i++)
    {
        ifile >> input[i];
    }

    // Pad inputs
    const int PADDED_SIZE = SAMPLES + 2*FILTER_LEN;
    const int OUTPUT_SIZE = SAMPLES + FILTER_LEN;

    float paddedInput[PADDED_SIZE];
    float output[OUTPUT_SIZE];
    float output_opt[OUTPUT_SIZE];
    float output_neon[OUTPUT_SIZE];

    for(int i = 0; i < PADDED_SIZE; i++)
    {
        if(i < FILTER_LEN || i >= SAMPLES+FILTER_LEN)
        {
            paddedInput[i] = 0;
        }
        else{ paddedInput[i] = input[i - FILTER_LEN]; }
    }

    // Measure baseline
    fir(coeffs, paddedInput, output, PADDED_SIZE, FILTER_LEN);

    // Measure optimized
    if(FILTER_LEN % 4 != 0){ return -1; }
    fir_opt(coeffs, paddedInput, output_opt, PADDED_SIZE, FILTER_LEN);

    if(FILTER_LEN % 4 != 0){ return -1; }
    fir_neon(coeffs, paddedInput, output_neon, PADDED_SIZE, FILTER_LEN);

    float mse = 0;
    float mse_opt = 0;
    float mse_neon = 0;
    float golden;

    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        gfile >> golden;
        float diff = output[i] - golden;
        mse += diff * diff;

        float diff_opt = output_opt[i] - golden;
        mse_opt += diff_opt * diff_opt;

        float diff_neon = output_neon[i] - golden;
        mse_neon += diff_neon * diff_neon;
    }

    printf("RMSE_naive: %.2f\n", sqrt(mse));
    printf("RMSE_opt:   %.2f\n", sqrt(mse_opt));
    printf("RMSE_neon:  %.2f\n", sqrt(mse_neon));

    return 0;
}
