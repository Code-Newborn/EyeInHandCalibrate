#include <iostream>
#include <fstream>
#include <sstream>
#include "EyeInHand.h"

void Tsai_HandEye(cv::Mat &Hcg, std::vector<cv::Mat> Hgij, std::vector<cv::Mat> Hcij)
{
	CV_Assert(Hgij.size() == Hcij.size()); // 判断输入数据的集合大小是否相等
	int nStatus = (int)Hgij.size();

	cv::Mat Rgij(3, 3, CV_64FC1);
	cv::Mat Rcij(3, 3, CV_64FC1);

	cv::Mat rgij(3, 1, CV_64FC1);
	cv::Mat rcij(3, 1, CV_64FC1);

	double theta_gij;
	double theta_cij;

	cv::Mat rngij(3, 1, CV_64FC1);
	cv::Mat rncij(3, 1, CV_64FC1);

	cv::Mat Pgij(3, 1, CV_64FC1);
	cv::Mat Pcij(3, 1, CV_64FC1);

	cv::Mat tempA(3, 3, CV_64FC1);
	cv::Mat tempb(3, 1, CV_64FC1);

	cv::Mat A;
	cv::Mat b;
	cv::Mat pinA;

	cv::Mat Pcg_prime(3, 1, CV_64FC1);
	cv::Mat Pcg(3, 1, CV_64FC1);
	cv::Mat PcgTrs(1, 3, CV_64FC1);

	cv::Mat Rcg(3, 3, CV_64FC1);
	cv::Mat eyeM = cv::Mat::eye(3, 3, CV_64FC1);

	cv::Mat Tgij(3, 1, CV_64FC1);
	cv::Mat Tcij(3, 1, CV_64FC1);

	cv::Mat tempAA(3, 3, CV_64FC1);
	cv::Mat tempbb(3, 1, CV_64FC1);

	cv::Mat AA;
	cv::Mat bb;
	cv::Mat pinAA;

	cv::Mat Tcg(3, 1, CV_64FC1);

	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i](cv::Rect(0, 0, 3, 3)).copyTo(Rgij); // 提取旋转矩阵
		Hcij[i](cv::Rect(0, 0, 3, 3)).copyTo(Rcij);

		Rodrigues(Rgij, rgij); // 旋转矩阵 转 旋转向量
		Rodrigues(Rcij, rcij);

		theta_gij = cv::norm(rgij); // 范数
		theta_cij = cv::norm(rcij);

		rngij = rgij / theta_gij; // 归一化，轴角表示
		rncij = rcij / theta_cij;

		Pgij = 2 * sin(theta_gij / 2) * rngij; // 修正的罗德里格斯参数表示姿态变化
		Pcij = 2 * sin(theta_cij / 2) * rncij;

		tempA = skew(Pgij + Pcij);
		tempb = Pcij - Pgij;

		A.push_back(tempA);
		b.push_back(tempb);
	}

	// 计算旋转矩阵
	invert(A, pinA, cv::DECOMP_SVD);

	Pcg_prime = pinA * b;
	Pcg = 2 * Pcg_prime / sqrt(1 + norm(Pcg_prime) * norm(Pcg_prime));
	PcgTrs = Pcg.t();
	Rcg = (1 - norm(Pcg) * norm(Pcg) / 2) * eyeM + 0.5 * (Pcg * PcgTrs + sqrt(4 - norm(Pcg) * norm(Pcg)) * skew(Pcg));

	// 计算平移矩阵
	for (int i = 0; i < nStatus; i++)
	{
		Hgij[i](cv::Rect(0, 0, 3, 3)).copyTo(Rgij);
		Hcij[i](cv::Rect(0, 0, 3, 3)).copyTo(Rcij);
		Hgij[i](cv::Rect(3, 0, 1, 3)).copyTo(Tgij);
		Hcij[i](cv::Rect(3, 0, 1, 3)).copyTo(Tcij);

		tempAA = Rgij - eyeM;
		tempbb = Rcg * Tcij - Tgij;

		AA.push_back(tempAA);
		bb.push_back(tempbb);
	}
	cv::invert(AA, pinAA, cv::DECOMP_SVD);
	Tcg = pinAA * bb;

	// 组装成齐次矩阵
	Rcg.copyTo(Hcg(cv::Rect(0, 0, 3, 3)));
	Tcg.copyTo(Hcg(cv::Rect(3, 0, 1, 3)));
	Hcg.at<double>(3, 0) = 0.0;
	Hcg.at<double>(3, 1) = 0.0;
	Hcg.at<double>(3, 2) = 0.0;
	Hcg.at<double>(3, 3) = 1.0;
}

cv::Mat skew(const cv::Mat A)
{
	CV_Assert(A.cols == 1 && A.rows == 3);
	cv::Mat B(3, 3, CV_64FC1);

	B.at<double>(0, 0) = 0.0;
	B.at<double>(0, 1) = -A.at<double>(2, 0);
	B.at<double>(0, 2) = A.at<double>(1, 0);

	B.at<double>(1, 0) = A.at<double>(2, 0);
	B.at<double>(1, 1) = 0.0;
	B.at<double>(1, 2) = -A.at<double>(0, 0);

	B.at<double>(2, 0) = -A.at<double>(1, 0);
	B.at<double>(2, 1) = A.at<double>(0, 0);
	B.at<double>(2, 2) = 0.0;

	return B;
}

cv::Mat DOF6ZYX_ToTransformMatrix(double x, double y, double z, double rx_deg, double ry_deg, double rz_deg)
{
	double alpha = rx_deg / (180 / CV_PI);
	double beta = ry_deg / (180 / CV_PI);
	double gama = rz_deg / (180 / CV_PI);

	cv::Mat TransformMatrix(4, 4, CV_64FC1);

	// ZYX 欧拉角转旋转矩阵
	double rotation[3][3] = {cos(beta) * cos(gama), sin(alpha) * sin(beta) * cos(gama) - cos(alpha) * sin(gama), cos(alpha) * sin(beta) * cos(gama) + sin(alpha) * sin(gama),
							 cos(beta) * sin(gama), sin(alpha) * sin(beta) * sin(gama) + cos(alpha) * cos(gama), cos(alpha) * sin(beta) * sin(gama) - sin(alpha) * cos(gama),
							 -sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta)};
	cv::Mat rotation_Matrix(3, 3, CV_64FC1, rotation);

	double translation[3][1] = {x, y, z};
	cv::Mat translation_matrix(3, 1, CV_64FC1, translation);

	rotation_Matrix.copyTo(TransformMatrix(cv::Rect(0, 0, 3, 3)));
	translation_matrix.copyTo(TransformMatrix(cv::Rect(3, 0, 1, 3)));

	TransformMatrix.at<double>(3, 0) = 0.0;
	TransformMatrix.at<double>(3, 1) = 0.0;
	TransformMatrix.at<double>(3, 2) = 0.0;
	TransformMatrix.at<double>(3, 3) = 1.0;

	return TransformMatrix;
}

bool TransformMatrix_ToDOF6ZYX(cv::Mat TransformMatrix, double &x, double &y, double &z, double &rx_deg, double &ry_deg, double &rz_deg)
{
	x = TransformMatrix.at<double>(0, 3);
	y = TransformMatrix.at<double>(1, 3);
	z = TransformMatrix.at<double>(2, 3);
	rx_deg = atan2(TransformMatrix.at<double>(2, 1), TransformMatrix.at<double>(2, 2)) * (180.0 / CV_PI);
	ry_deg = atan2(-TransformMatrix.at<double>(2, 0), sqrt(1 - pow(TransformMatrix.at<double>(2, 0), 2))) * (180.0 / CV_PI);
	rz_deg = atan2(TransformMatrix.at<double>(1, 0), TransformMatrix.at<double>(0, 0)) * (180.0 / CV_PI);
	return true;
}

bool ReadTxt(std::string filepath, int col, std::vector<double> &data)
{

	std::ifstream ifs; //创建流对象

	ifs.open(filepath, std::ios::in); //打开文件

	if (!ifs.is_open()) //判断文件是否打开
	{
		std::cout << "打开文件失败！！！";
		return false;
	}

	std::vector<std::string> item; // 用于存放文件中的每一行字符串数据

	std::string temp; // 把文件中的一行数据作为字符串放入容器中

	while (getline(ifs, temp)) // 利用getline()读取每一行文本数据，存放在 vector 中
	{
		item.push_back(temp);
	}

	// 处理每一行的文本数据
	for (auto it = item.begin(); it != item.end(); it++)
	{
		std::istringstream istr(*it); //其作用是把字符串分解为单词(在此处就是把一行数据分为单个数据)

		std::string str;

		int count = 0; //统计一行数据中单个数据个数

		//获取文件中的第 1、2 列数据
		while (istr >> str) //以空格为界，把istringstream中数据取出放入到依次s中
		{
			//获取第1列数据
			if (count == col)
			{
				double r = atof(str.c_str()); //数据类型转换，将string类型转换成double,如果字符串不是由数字组成，则字符被转化为 0

				data.push_back(r);
			}
			count++;
		}
	}
	return true;
}

bool ErrorCalculation_EyeInHand(std::vector<cv::Mat> A, cv::Mat T_G2C, std::vector<cv::Mat> B, EyeInHand_ErrorEstimatation &error)
{

	CV_Assert(A.size() == B.size()); // 判断输入数据的集合大小是否相等

	for (int i = 0; i < A.size(); i++)
	{
		double rectify_x, rectify_y, rectify_z, rectify_rx, rectify_ry, rectify_rz;
		double measure_x, measure_y, measure_z, measure_rx, measure_ry, measure_rz;
		cv::Mat T_G2C_inv;
		cv::invert(T_G2C, T_G2C_inv);
		cv::Mat rectify_A = T_G2C * B[i] * T_G2C_inv;
		TransformMatrix_ToDOF6ZYX(rectify_A, rectify_x, rectify_y, rectify_z, rectify_rx, rectify_ry, rectify_rz); // 校正数据 转欧拉角ZYX位姿数据
		TransformMatrix_ToDOF6ZYX(A[i], measure_x, measure_y, measure_z, measure_rx, measure_ry, measure_rz);	   // 测量数据 转欧拉角ZYX位姿数据
		error.X_error.push_back(measure_x - rectify_x);															   // 记录误差值
		error.Y_error.push_back(measure_y - rectify_x);
		error.Z_error.push_back(measure_z - rectify_x);
		error.RX_error.push_back(measure_rx - rectify_rx);
		error.RY_error.push_back(measure_ry - rectify_ry);
		error.RZ_error.push_back(measure_rz - rectify_rz);
	}

	error.X_error_SampleStdDeviation = RootMeanSquare(error.X_error);
	error.Y_error_SampleStdDeviation = RootMeanSquare(error.Y_error);
	error.Z_error_SampleStdDeviation = RootMeanSquare(error.Z_error);
	error.RX_error_SampleStdDeviation = RootMeanSquare(error.RX_error);
	error.RY_error_SampleStdDeviation = RootMeanSquare(error.RY_error);
	error.RZ_error_SampleStdDeviation = RootMeanSquare(error.RZ_error);

	return true;
}

double SampleStdDeviation(std::vector<double> data)
{
	double sum = 0;
	double average = 0;

	double temp_sum = 0;
	double sample_stdDeviation = 0;

	int n = (int)data.size();

	for (auto iterator = data.begin(); iterator != data.end(); iterator++)
	{
		sum += *iterator;
	}
	average = sum / n;

	for (auto iterator = data.begin(); iterator != data.end(); iterator++)
	{
		temp_sum += pow((*iterator - average), 2);
	}
	sample_stdDeviation = sqrt(temp_sum / (n - 1));

	return sample_stdDeviation;
}

double RootMeanSquare(std::vector<double> data)
{
	double sum = 0;

	double root_mean_square = 0;

	int n = (int)data.size();

	for (auto iterator = data.begin(); iterator != data.end(); iterator++)
	{
		sum += pow(*iterator, 2);
	}
	root_mean_square = sqrt(sum / n);
	return root_mean_square;
}