#include <iostream>
#include <stdio.h>
#include <math.h>

#include "mat.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "cycletime.h"

int main(int argc, const char* argv[])
{
    uint64_t cpu_count1 = 0;
    uint64_t cpu_count2 = 0;;
    uint64_t diff = 0;

    init_counters(1, 0);
    uint16_t row = 12;
    uint16_t col = 12;
    Matrix<float> m1(row, col);
    Matrix<float> m2(row,col);

    for(uint16_t i =0; i< row; i++)
    {
        for(uint16_t j=0;j< col; j++)
        {
            m1.put(i,j,(i+1));
            m2.put(i,j,(i+1));
        }
    }

    //m1.print_matrix();
    //m2.print_matrix();
    //Matrix<float>m3;
    cpu_count1 = get_cyclecount();
    Matrix<float>m3 = m1*m2; // as per requirement.
    cpu_count2 = get_cyclecount();
    diff = cpu_count2 - cpu_count1;
    std::cout << "CPU Count difference Naive approach= "<< diff << std::endl;
    //m3.print_matrix();
#if 0
    for(int i = 0; i < m3.get_matrix_height(); i++)
    {
        for(int j = 0; j<m3.get_matrix_width(); j++)
        {
            m3.add(i,j,1);
        }
    }
#endif
#if 0
    /***Block size****/
    uint16_t block_size = 4; 
    cpu_count1 = get_cyclecount();
    Matrix<float>m4 = block_matrix_multiply(m1,m2,block_size);
    cpu_count2 = get_cyclecount();
    diff = cpu_count2 - cpu_count1;
    std::cout << "CPU Count difference BMM approach= "<< diff << std::endl;
    //m4.print_matrix();
//#if 0
    for(int i = 0; i < m3.get_matrix_height(); i++)
    {
        for(int j = 0; j<m3.get_matrix_width(); j++)
        {
            float m3val = m3.get(i,j);
            float m4val = m4.get(i,j);
            if(m3val!=m4val)
            {
                std::cout << "Matrix m3 = " << m3val << "Matrix m4 = " << m4val << std::endl;
            }
        }
    }
#endif
    //Comparison with Eigen Matrix;
    Eigen::MatrixXf eig_mat_a(row,col);
    Eigen::MatrixXf eig_mat_b(row,col);

    for(uint16_t i =0; i< row; i++)
    {
        for(uint16_t j=0;j< col; j++)
        {
            eig_mat_a(i,j) = i+1;
            eig_mat_b(i,j) = i+1;
        }
    }
    cpu_count1 = get_cyclecount();
    Eigen::MatrixXf eig_mat_c = eig_mat_a*eig_mat_b;
    cpu_count2 = get_cyclecount();
    diff = cpu_count2 - cpu_count1;
    std::cout << "CPU Count difference Eigen Lib= "<< diff << std::endl;
    //std::cout << "Eigen Matrix\n" << eig_mat_c;

    //Comparison with OpenCV
    cv::Mat ocv_a = cv:: Mat::zeros(row, col, CV_32FC1); 
    cv::Mat ocv_b = cv:: Mat::zeros(row, col, CV_32FC1);

    for(uint16_t i =0; i< row; i++)
    {
        for(uint16_t j=0;j< col; j++)
        {
            ocv_a.at<float>(i,j) = i+1;
            ocv_b.at<float>(i,j) = i+1;
        }
    }
    cpu_count1 = get_cyclecount();
    cv::Mat ocv_c = ocv_a*ocv_b;
    cpu_count2 = get_cyclecount();
    diff = cpu_count2 - cpu_count1;
    std::cout << "CPU Count difference OpenCV Lib= "<< diff << std::endl;
    //std::cout << "Open CV Matrix\n" << ocv_c;
    return 0;
}
