#include "opencv2/opencv.hpp"

/***
 * @name: 反对称矩阵
 * @msg: 计算反对称矩阵
 * @param {Mat} A 输入 3*1 矩阵
 * @return {Mat} 反对称矩阵 3*3矩阵
 */
cv::Mat skew(const cv::Mat A);

/***
 * @name: 手眼标定（EyeInHand 眼在手上）
 * @msg: 参考 https://blog.csdn.net/qq_29462849/article/details/118159686
 * @param {Mat} &Hcg 输出 靶标坐标系相对于执行器末端的位姿描述
 * @param {vector<cv::Mat>} Hgij 执行器末端坐标系之间相对位置姿态的齐次变换矩阵 集合
 * @param {vector<cv::Mat>} Hcij 摄像机坐标系之间相对位置姿态的齐次变换矩阵 集合
 * @return {*}
 */
void Tsai_HandEye(cv::Mat &Hcg, std::vector<cv::Mat> Hgij, std::vector<cv::Mat> Hcij);

/*** 
 * @name: 6自由度位姿数据转齐次矩阵
 * @msg: 欧拉角 ZYX 顺序
 * @param {double} x 平移量X
 * @param {double} y 平移量Y
 * @param {double} z 平移量Z
 * @param {double} euler_x 欧拉角RX
 * @param {double} euler_y 欧拉角RY
 * @param {double} euler_z 欧拉角RZ
 * @return {*}
 */
cv::Mat DOF6ZYX_ToTransformMatrix(double x, double y, double z, double euler_x, double euler_y, double euler_z);

/*** 
 * @name: 读取 txt 文件某列数据 以空格分隔
 * @msg: 
 * @param {string} filepath 文件路径
 * @param {int} col 读取的列索引，0开始
 * @param {vector<double>} &data 输出数据
 * @return {*}
 */
bool ReadTxt(std::string filepath, int col,std::vector<double> &data);