#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <random>
// #include "opencv2/viz.hpp"
#include <algorithm>
using namespace std;
void add_print(int a, int b);


cv::Point2f pixel2cam(const cv::Point2f& point,const cv::Mat& K);

Eigen::Vector3f pixel2eigen(const cv::Point2f& p,const cv::Mat& K);

void drawReprojection(cv::Mat& img, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);

cv::Point2f point2pixel(cv::Mat point, cv::Mat K, cv::Mat R, cv::Mat t);

void points2pixelVector(std::vector<Eigen::Vector3f>& points, std::vector<cv::Point2f>& campoint, const cv::Mat K, const cv::Mat R, const cv::Mat t);

void reprojectionErrorStatistics(std::vector<cv::Point2f>& uv1, std::vector<cv::Point2f>& uv2);

void drawX(cv::Mat& image, const cv::Point& point, int size, const cv::Scalar& color, int thickness);


void saveDataToFile(const std::vector<cv::Point2f>& data, const std::string& filename);


void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

Eigen::MatrixXf cvMatToEigen(const cv::Mat& cvMat);



bool readAndInverse(string figure_path);

#endif