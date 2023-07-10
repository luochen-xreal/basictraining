#ifndef SOLVEPOSE_H
#define SOLVEPOSE_H

#include "include/utils.h"
#include "include/triangulation.h"
void solvePose(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat CameraMatrix, Eigen::Matrix3f& R_,
                Eigen::Vector3f& t_, Eigen::Matrix3f& essential_matrix);

void solvePoseOpenCV(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier,
                std::vector<cv::DMatch>& matches, 
                const cv::Mat& CameraMatrix, cv::Mat& R, cv::Mat& t, cv::Mat& essential_matrix);


int recoverPose(const Eigen::Matrix3f E,const std::vector<Eigen::Vector3f>& _points1,const std::vector<Eigen::Vector3f>& _points2, const Eigen::Matrix3f& _cameraMatrix,
                    Eigen::Matrix3f& _R,
                    Eigen::Vector3f& _t);
#endif