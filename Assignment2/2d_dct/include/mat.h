#ifndef __MAT_H_
#define __MAT_H_
#include <iostream>
#include <vector>

enum matrix_result
{
    MAT_FAILURE = -2,
    MAT_FAILURE_MEM, 
    MAT_RESULT_SUCCESS,
    MAT_RESULT_MAX
};

template <class T>
class Matrix
{
    /** private members, size and containers */
private:
    std::vector<std::vector<T>> Mat; //Two dimensional vector
    uint16_t height;
    uint16_t width;

public://place holder of constructor destructor
    Matrix<T>(uint16_t height, uint16_t width);
    Matrix<T>(std::vector<std::vector<T>> const &input_mat);
    Matrix<T>();

public: // place holder for special capabilities of matrix
    void     transpose(Matrix<T>&)const;

    public: // place holder to general capabilities of matrix
    uint16_t get_matrix_height() const; //adding const would make this function read only
    uint16_t get_matrix_width() const;
    void     put(uint16_t row, uint16_t col, T value);
    T        get(uint16_t row, uint16_t col) const;
    void     add(uint16_t row, uint16_t col, T val);
    void     print_matrix();
};

template <class T> 
Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b);

template<class T>
Matrix<T> block_matrix_multiply(const Matrix<T>&a, const Matrix<T>&b, uint16_t block_size);

#endif
