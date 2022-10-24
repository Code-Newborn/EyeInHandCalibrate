#ifndef _EYEINHAND_H_
#define _EYEINHAND_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include "opencv2/opencv.hpp"

struct EyeInHand_ErrorEstimatation
{
    // 误差记录
    std::vector<double> X_error;
    std::vector<double> Y_error;
    std::vector<double> Z_error;
    std::vector<double> RX_error;
    std::vector<double> RY_error;
    std::vector<double> RZ_error;

    // 样本标准差
    double X_error_SampleStdDeviation = 0;
    double Y_error_SampleStdDeviation = 0;
    double Z_error_SampleStdDeviation = 0;

    // 样本方差
    double RX_error_SampleStdDeviation = 0;
    double RY_error_SampleStdDeviation = 0;
    double RZ_error_SampleStdDeviation = 0;
};

/// @brief      反对称矩阵转换函数
/// @msg        计算反对称矩阵
/// @param      {Mat} input_Matrix 输入 3*1 矩阵
/// @return     {Mat} 反对称矩阵 3*3矩阵
cv::Mat skew(const cv::Mat input_Matrix);

/// @brief      手眼标定函数（EyeInHand 眼在手上）
/// @msg        Tsai方法 参考：https://blog.csdn.net/qq_29462849/article/details/118159686
/// @param      {Mat} &Hcg 靶标坐标系相对于执行器末端的位姿描述
/// @param      {vector<cv::Mat>} Hgij 执行器末端坐标系之间相对位置姿态的齐次变换矩阵 集合
/// @param      {vector<cv::Mat>} Hcij 摄像机坐标系之间相对位置姿态的齐次变换矩阵 集合
/// @return     {*}
void Tsai_HandEye(cv::Mat &Hcg, std::vector<cv::Mat> Hgij, std::vector<cv::Mat> Hcij);

/// @brief      6自由度位姿数据转齐次矩阵函数
/// @msg        欧拉角 ZYX 顺序
/// @param      {double} x 平移量X
/// @param      {double} y 平移量Y
/// @param      {double} z 平移量Z
/// @param      {double} euler_x 欧拉角RX
/// @param      {double} euler_y 欧拉角RY
/// @param      {double} euler_z 欧拉角RZ
/// @return     {*}
cv::Mat DOF6ZYX_ToTransformMatrix(double x, double y, double z, double euler_x, double euler_y, double euler_z);

/// @brief      齐次矩阵转6自由度位姿数据函数
/// @msg        依据齐次变换矩阵计算欧拉角顺序ZYX的6自由度位姿数据
/// @param      {Mat} TransformMatrix 齐次变换矩阵
/// @param      {double} &x 平移量X
/// @param      {double} &y 平移量Y
/// @param      {double} &z 平移量Z
/// @param      {double} &rx_deg 欧拉角RX（单位角度）
/// @param      {double} &ry_deg 欧拉角RY（单位角度）
/// @param      {double} &rz_deg 欧拉角RZ（单位角度）
/// @return     {bool}
bool TransformMatrix_ToDOF6ZYX(cv::Mat TransformMatrix, double &x, double &y, double &z, double &rx_deg, double &ry_deg, double &rz_deg);

/// @brief      读取 txt 文件某列数据
/// @msg        文件内容以空格分隔列
/// @param      {string} filepath 文件路径
/// @param      {int} col 取的列索引，0开始
/// @param      {vector<double>} &data 输出数据
/// @return     {bool}
bool ReadTxt(std::string filepath, int col, std::vector<double> &data);

/// @brief      重投影矩阵误差计算
/// @msg
/// @param      {vector<cv::Mat>} T_B2Gs 基座To执行末端的齐次变换阵集合
/// @param      {Mat} T_G2C 标定的执行末端到相机的齐次变换矩阵结果
/// @param      {vector<cv::Mat>} T_C2Os 相机To靶标的齐次变换阵集合
/// @param      {EyeInHand_ErrorEstimatation} &error 存储误差的结构体
/// @return     {bool}
bool ErrorCalculation_EyeInHand(std::vector<cv::Mat> T_B2Gs, cv::Mat T_G2C, std::vector<cv::Mat> T_C2Os, EyeInHand_ErrorEstimatation &error);

/// @brief      样本标准差计算
/// @msg        用于评估数据的离散分布情况
/// @param      {vector<double>} data 输入数据
/// @return     {double} 标准差值
double SampleStdDeviation(std::vector<double> data);

/// @brief      均方根计算
/// @msg        计算有效值，一般用于分析噪声
/// @param      {vector<double>} data 输入数据
/// @return     {double} 均方根值
double RootMeanSquare(std::vector<double> data);

#endif