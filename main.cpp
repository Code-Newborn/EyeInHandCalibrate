#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <queue>
#include "EyeInHand.h"

int main(int, char **)
{
    std::vector<double> data_b2g_x;
    std::vector<double> data_b2g_y;
    std::vector<double> data_b2g_z;
    std::vector<double> data_b2g_rx;
    std::vector<double> data_b2g_ry;
    std::vector<double> data_b2g_rz;
    ReadTxt("../T_b2g.txt", 0, data_b2g_x);
    ReadTxt("../T_b2g.txt", 1, data_b2g_y);
    ReadTxt("../T_b2g.txt", 2, data_b2g_z);
    ReadTxt("../T_b2g.txt", 3, data_b2g_rx);
    ReadTxt("../T_b2g.txt", 4, data_b2g_ry);
    ReadTxt("../T_b2g.txt", 5, data_b2g_rz); // 读取位姿数据
    std::vector<double> data_c2o_x;
    std::vector<double> data_c2o_y;
    std::vector<double> data_c2o_z;
    std::vector<double> data_c2o_rx;
    std::vector<double> data_c2o_ry;
    std::vector<double> data_c2o_rz;
    ReadTxt("../T_c2o.txt", 0, data_c2o_x);
    ReadTxt("../T_c2o.txt", 1, data_c2o_y);
    ReadTxt("../T_c2o.txt", 2, data_c2o_z);
    ReadTxt("../T_c2o.txt", 3, data_c2o_rx);
    ReadTxt("../T_c2o.txt", 4, data_c2o_ry);
    ReadTxt("../T_c2o.txt", 5, data_c2o_rz);

    std::vector<cv::Mat> T_A;
    std::vector<cv::Mat> T_B;

    if (data_b2g_x.size() == data_b2g_y.size() && data_b2g_x.size() == data_b2g_z.size() && // 判断数据是否对齐
        data_b2g_x.size() == data_b2g_rx.size() && data_b2g_ry.size() == data_b2g_rz.size() &&
        data_b2g_x.size() == data_c2o_x.size() && data_c2o_y.size() == data_c2o_z.size() &&
        data_b2g_x.size() == data_c2o_rx.size() && data_c2o_ry.size() == data_c2o_rz.size())
    {
        for (int i = 0; i < data_b2g_x.size(); i++)
        {
            cv::Mat T_b2g;
            cv::Mat T_c2o;
            T_b2g = DOF6ZYX_ToTransformMatrix(data_b2g_x.at(i), data_b2g_y.at(i), data_b2g_z.at(i), data_b2g_rx.at(i), data_b2g_ry.at(i), data_b2g_rz.at(i));
            T_c2o = DOF6ZYX_ToTransformMatrix(data_c2o_x.at(i), data_c2o_y.at(i), data_c2o_z.at(i), data_c2o_rx.at(i), data_c2o_ry.at(i), data_c2o_rz.at(i));

            T_A.push_back(T_b2g);
            T_B.push_back(T_c2o);
        }
    }
    else
    {
        std::cout << "data in file is not align" << std::endl;
    }

    // 生成 AX = XB中的 A
    std::vector<cv::Mat> A;
    std::vector<cv::Mat> B;
    for (int i = 1; i < T_A.size(); i++)
    {
        cv::Mat temp_invA;
        cv::Mat temp_invB;
        cv::Mat item_A;
        cv::Mat item_B;
        cv::invert(T_A.at(i), temp_invA);
        item_A = temp_invA * T_A.at(i - 1);
        A.push_back(item_A);
        cv::invert(T_B.at(i - 1), temp_invB);
        item_B = T_B.at(i) * temp_invB;
        B.push_back(item_B);
    }

    std::cout << "==================== A Matrix ====================" << std::endl;
    for (int i = 0; i < A.size(); i++)
    {
        std::cout << A.at(i) << std::endl;
    }
    std::cout << "==================== B Matrix ====================" << std::endl;
    for (int i = 0; i < B.size(); i++)
    {
        std::cout << B.at(i) << std::endl;
    }

    std::cout << "==================== X Result ====================" << std::endl;

    cv::Mat X_result(4, 4, CV_64FC1);
    Tsai_HandEye(X_result, A, B);
    std::cout << X_result << std::endl;

    std::cout << "==================== Error ====================" << std::endl;

    // 手眼标定结果
    double _T_G2C[4][4] = {0.2049888678171762, -0.07546780275972509, -0.9758504879425185, -789.0921101103351,
                           0.9763668686332763, -0.053974064078531, 0.209271446313388, 86.67447300617494,
                           -0.0684638730008964, -0.9956864020129359, 0.06262017997619668, 114.8740101957056,
                           0, 0, 0, 1};
    cv::Mat T_G2C(4, 4, CV_64FC1, _T_G2C);
    EyeInHand_ErrorEstimatation error;

    if (ErrorCalculation_EyeInHand(A, T_G2C, B, error))
    {
        std::cout << "Sample Standard Deviation X :" << error.X_error_SampleStdDeviation << std::endl;
        std::cout << "Sample Standard Deviation Y :" << error.Y_error_SampleStdDeviation << std::endl;
        std::cout << "Sample Standard Deviation Z :" << error.Z_error_SampleStdDeviation << std::endl;
        std::cout << "Sample Standard Deviation RX :" << error.RX_error_SampleStdDeviation << std::endl;
        std::cout << "Sample Standard Deviation RY :" << error.RY_error_SampleStdDeviation << std::endl;
        std::cout << "Sample Standard Deviation RZ :" << error.RZ_error_SampleStdDeviation << std::endl;
    }

    std::cout << " Translation Error :"
              << sqrt(pow(error.X_error_SampleStdDeviation, 2) + pow(error.Y_error_SampleStdDeviation, 2) + pow(error.Z_error_SampleStdDeviation, 2))
              << std::endl;
    ;

    return 0;
}
