#include <iostream>
#include <opencv2/opencv.hpp>


void add_print(int a, int b);


void solvePose(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& matches, const cv::Mat CameraMatrix);