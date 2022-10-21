#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <queue>
#include "EyeInHand.h"

int main(int, char **)
{
    std::cout << "Hello, world!\n";

    double a1[4][4] = {-0.9976, 0.0676, -0.0173, -146.8929,
                       -0.0697, -0.9488, 0.3082, -43.6165,
                       0.0044, 0.3087, 0.9512, 10.3295,
                       0, 0, 0, 1};
    double a2[4][4] = {0.0535, -0.7980, 0.6003, -165.7422,
                       0.9483, 0.2289, 0.2197, 44.2528,
                       -0.3127, 0.5575, 0.7690, 21.0485,
                       0, 0, 0, 1};
    double b1[4][4] = {-0.9975, 0.0711, -0.0029, -11.6622,
                       -0.0663, -0.9443, -0.3223, 104.2345,
                       -0.0257, -0.3213, 0.9466, 21.3907,
                       0, 0, 0, 1};
    double b2[4][4] = {0.0544, -0.7855, -0.6164, 177.7842,
                       0.9515, 0.2280, -0.2067, 65.4536,
                       0.3029, -0.5753, 0.7598, 84.5619,
                       0, 0, 0, 1};
    cv::Mat A1(4, 4, CV_64FC1, a1);
    cv::Mat A2(4, 4, CV_64FC1, a2);
    cv::Mat B1(4, 4, CV_64FC1, b1);
    cv::Mat B2(4, 4, CV_64FC1, b2);
    std::vector<cv::Mat> Hgij;
    std::vector<cv::Mat> Hcij;
    Hgij.push_back(A1);
    Hgij.push_back(A2);
    Hcij.push_back(B1);
    Hcij.push_back(B2);
    cv::Mat Hcg1(4, 4, CV_64FC1);
    Tsai_HandEye(Hcg1, Hgij, Hcij);
    // std::cout << Hcg1 << std::endl;

    double _test1[4][4] = {1, 0, 0, 0,
                           0, 0, -0, 10,
                           -0, 0.8660254037844387, -0.4999999999999998, 20,
                           0, 0, 0, 1};

    double _test2[4][4] = {-0.4999999999999998, -0.8660254037844387, 0, 0,
                           0.8660254037844387, 0.8660254037844387, -0, 0,
                           -0, 0, 1, 0,
                           0, 0, 0, 1};

    cv::Mat test1(4, 4, CV_64FC1, _test1);
    cv::Mat test2(4, 4, CV_64FC1, _test2);
    cv::Mat test;

    cv::Mat test1_inv;
    cv::invert(test1, test1_inv);
    std::cout << test1_inv << std::endl;
    test = test1_inv * test2;

    std::cout << test << std::endl;
    std::cout << "==================== Split ====================" << std::endl;

    std::vector<double>
        data_b2g_x;
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

    for (int i = 0; i < T_A.size(); i++)
    {
        std::cout << T_A.at(i) << std::endl;
    }
    std::cout << "==================== Split ====================" << std::endl;
    for (int i = 0; i < T_B.size(); i++)
    {
        std::cout << T_B.at(i) << std::endl;
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

    return 0;
}
