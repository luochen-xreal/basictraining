#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <include/utils.h>

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,
                const cv::Mat CameraMatrix, const cv::Mat& R, const cv::Mat& t, std::vector<Eigen::Vector3f> &points);

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,  
                const cv::Mat CameraMatrix, const Eigen::Matrix3f& R, const Eigen::Vector3f& t, vector<Eigen::Vector3f> &points);

void Triangulate(const Eigen::MatrixXf &P1, const Eigen::MatrixXf &P2, const Eigen::Vector3f &kp1, const Eigen::Vector3f &kp2, 
Eigen::Vector4f &x3D);

#endif