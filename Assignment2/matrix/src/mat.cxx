#include <iostream>
#include <stdio.h>
#include <math.h>
#include "mat.h"

template <class T>
Matrix<T>::Matrix(uint16_t height, uint16_t width)
{
    this->height = height;
    this->width = width;
    this->Mat = std::vector< std::vector<T> >(height, std::vector<T>(width));
}

template <class T>
Matrix<T>::Matrix(std::vector<std::vector<T>> const &input_mat)
{
    if(0 == input_mat.size())
    {
        std::cerr<< "Input vector should be greater that 0" << std::endl;
        return;
    }
    if(input_mat.size() != input_mat[0].size())
    {
        std::cerr<< "Not a square Matrix" << std::endl;
        return;
    }

    this->height = input_mat.size();
    this->width = input_mat[0].size();
    this->Mat = input_mat;
}

template<class T>
Matrix<T>::Matrix()
{
    this->height = 0;
    this->width = 0; 
}

template<class T>
uint16_t Matrix<T>::get_matrix_height() const
{
    return this->height;
}

template<class T>
uint16_t Matrix<T>::get_matrix_width() const
{
    return this->width;
}

template<class T>
void Matrix<T>::put(uint16_t row, uint16_t col, T value)
{
    if(row >= this->height || col >= this->width)
    {
        std::cerr<< "Invalid row and columns" << std:: endl;
    }
    else
    {
        this->Mat[row][col] = value;
    }
}

template<class T>
T Matrix<T>::get(uint16_t row, uint16_t col)const
{
    if(row >= this->height || col >= this->width )
    {
        std::cerr<< "Invalid row and columns" << std:: endl;
        return MAT_FAILURE;
    }
    else
    {
        return this->Mat[row][col];
    }
}

template<class T>
void Matrix<T>::transpose(Matrix<T>& mat)const
{
    if(mat.height && mat.width && mat.height == this->width && mat.width == this->height)
    {
        typename std::vector<T>::iterator ita_col;
        typename std::vector<std::vector <T>>::iterator ita_row;

        for(ita_row = mat.Mat.begin(); ita_row != mat.Mat.end(); ita_row++)
        {
            for(ita_col = mat.Mat[ita_row].begin(); ita_col != mat.Mat[ita_row].end(); ita_col++)
            {

                mat.Mat[ita_row][ita_col] = this->Mat[ita_col][ita_row];
            }
        }

    }
    else
    {
        std::cerr<< "Improper Matrix" << std::endl;
    }
}

template <class T> 
Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
{
    Matrix<T> c(a.height, b.width);
    T temp = 0;;
    if(a.height && a.width && b.height && b.width && a.width == b.height)
    {
        for(int i = 0; i < a.height(); i++)
        {  
            for(int j = 0; j < b.width(); j++)
            {
                for(int k = 0; k< a.width(); k++)
                {
                    temp += a.Mat[i][k] * b.Mat[k][j];
                }
                c.Mat[i][j] = temp; 
                temp = 0;
            }
        }
    }
    else
    {
        std::cerr<<"Not a valid matrix multiplication" << std::endl;
    }
    return c;
}
//used: youtube channel : compilerai to understand the block multiplication and used the code segment shown as the logic.
// It makes use of partial addition of dot products. i follows ii to ii+b, j follows jj to hh+b, k follows in direction of i and from
//kk to kk+b. With this method, we do not need to iterate over the whole height/width to get total dot product.
//it keeps accumulating the dot products. It improves spatial locality.
template<class T>
Matrix<T> block_matrix_multiply(const Matrix<T>&a, const Matrix<T>&b, uint16_t block_size)
{
    uint16_t size_mat    = 0;
    Matrix<T> c(a.height, b.width);
    T temp = 0;
    if(a.height && a.width && b.height && b.width && a.width == b.height && a.height == b.width && a.height == a.width)
    {
        size_mat = a.height;
        for(int ii = 0 ; ii < size_mat; ii += block_size)
        {   
            for(int jj = 0; jj < size_mat; jj +=block_size)
            {
                for(int kk = 0; kk < size_mat; kk += block_size)
                {
                    for(int i = ii; i< (ii+block_size > size_mat?size_mat:ii+block_size);i++)
                    {
                        for(int j = jj; j< (jj+block_size > size_mat?size_mat:ii+block_size); j++)
                        {
                            for(int k = kk; k< (kk+block_size > size_mat?size_mat:ii+block_size); k++)
                            {
                                c.Mat[i,j] += a.Mat[i, k]*b.Mat[k,j];
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        std::cerr << "Incorrect Matrix dimensions" << std::endl;
    }

    return c;
}