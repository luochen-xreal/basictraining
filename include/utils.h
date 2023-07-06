#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void add_print(int a, int b);

void solvePoseOpenCV(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat& CameraMatrix, cv::Mat& R, cv::Mat& t, cv::Mat& essential_matrix);

void solvePose(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& matches, const cv::Mat CameraMatrix);

cv::Point2f pixel2cam(const cv::Point2f& point,const cv::Mat& K);

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat CameraMatrix, const cv::Mat& R, const cv::Mat& t, std::vector<Eigen::Vector3d> &points);


