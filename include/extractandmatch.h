#ifndef EXTRACTANDMATCH_H
#define EXTRACTANDMATCH_H


#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>



void featureMatchingRANSAC(cv::Mat& img1, cv::Mat& img2,
                           std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matchesRANSAC,
                           int featureCount, double ransacThreshold);

#endif