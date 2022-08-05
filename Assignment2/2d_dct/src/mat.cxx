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
        for (int i=0 ; i<width ; i++)
        {
            for (int j=0 ; j<height ; j++)
            {
                mat.Mat[i][j] = this->Mat[j][i];
            }
        }

    }
    else
    {
        std::cerr<< "Improper Matrix" << std::endl;
    }
}

template<class T>
void Matrix<T>::add(uint16_t row, uint16_t col, T val)
{
    this->Mat[row][col] += val;
}

template <class T>
void Matrix<T>:: print_matrix()
{
    std::cout <<"------------------------------------" << std::endl;
    std:: cout << "Matrix " << this << std::endl;
    for (int i=0 ; i<width ; i++)
    {
        for (int j=0 ; j<height ; j++)
        {
            std::cout << this->Mat[i][j]<< "    ";
        }
        std::cout << std::endl;
    }
    std::cout <<"------------------------------------" << std::endl;
}

template <class T> 
Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
{
    uint16_t height_a = a.get_matrix_height();
    uint16_t width_a  = a.get_matrix_width();
    uint16_t height_b = b.get_matrix_height();
    uint16_t width_b  = b.get_matrix_width();

    Matrix<T> c(height_a, width_b);
    T temp = 0;;
    if(height_a && width_a && height_b && width_b && width_a == height_b)
    {
        for(int i = 0; i < height_a; i++)
        {  
            for(int j = 0; j < width_b; j++)
            {
                for(int k = 0; k< width_a; k++)
                {
                    temp += a.get(i,k)*b.get(k,j);
                }
                c.put(i,j,temp);
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
    uint16_t height_a = a.get_matrix_height();
    uint16_t width_a  = a.get_matrix_width();
    uint16_t height_b = b.get_matrix_height();
    uint16_t width_b  = b.get_matrix_width();
    
    Matrix<T> c(height_a, width_b);
    T temp = 0;
    if(height_a && width_a && height_b && width_b && width_a == height_b && height_a == width_b && height_a == width_a)
    {
        size_mat = height_a;
        for(int ii = 0 ; ii < size_mat; ii += block_size)
        {   
            for(int jj = 0; jj < size_mat; jj +=block_size)
            {
                for(int kk = 0; kk < size_mat; kk += block_size)
                {
                    //for(int i = ii; i< (ii+block_size > size_mat?size_mat:ii+block_size);i++)
                    for(int i = ii; i< (ii+block_size )/*> size_mat?size_mat:ii+block_size)*/;i++)
                    {
                        //for(int j = jj; j< (jj+block_size > size_mat?size_mat:ii+block_size); j++)
                        for(int j = jj; j< (jj+block_size)/* > size_mat?size_mat:ii+block_size)*/; j++)
                        {
                            //for(int k = kk; k< (kk+block_size > size_mat?size_mat:ii+block_size); k++)
                            for(int k = kk; k< (kk+block_size)/* > size_mat?size_mat:ii+block_size)*/; k++)
                            {
                                c.add(i,j,a.get(i,k)*b.get(k,j));
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

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;
template class Matrix<uint8_t>;
template class Matrix<uint16_t>;
template class Matrix<uint32_t>;
template class Matrix<uint64_t>;

template class Matrix<float> operator*<float>(Matrix<float> const&, Matrix<float> const&);
template class Matrix<float> block_matrix_multiply(const Matrix<float>&, const Matrix<float>&, uint16_t);



