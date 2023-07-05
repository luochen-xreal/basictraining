
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "include/TestFunction.h"
#include "include/ExtractAndMatch.h"
using namespace std;

bool readAndInverse(string figure_path){
    cv::Mat image = cv::imread(figure_path + ".png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "无法读取图片" << std::endl;
        return false;
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelValue = image.at<uchar>(i, j);
            image.at<uchar>(i, j) = 255 - pixelValue;
        }
    }

    // 可视化结果
    cv::imshow("Inverted Image", image);
    cv::waitKey(0);

    // 保存结果
    cv::imwrite(figure_path + "_output.png", image);
    return true;
}

struct camera_param{
    
    std::string sensorType;
    std::string comment;
    cv::Mat T_BS;
    int rateHz;
    vector<int> resolution;
    std::string cameraModel;
    vector<double> intrinsics;
    std::string distortionModel;
    vector<double> distortionCoefficients;
};

void ReadParameter(camera_param& param, string yaml){
    
    cout << "yaml:" << yaml << endl;
    cv::FileStorage fs(yaml, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cout << "无法打开YAML文件" << std::endl;
        return;
    }
    // 读取数据
    // fs["sensor_type"] >> param.sensorType;
    // fs["comment"] >> param.comment;
    cv::Mat T;
    fs["T_BS"] >> T;
    // fs["rate_hz"] >> param.rateHz;
    fs["resolution"] >> param.resolution;
    // fs["camera_model"] >> param.cameraModel;
    fs["intrinsics"] >> param.intrinsics;
    // fs["distortion_model"] >> param.distortionModel;
    fs["distortion_coefficients"] >> param.distortionCoefficients;
    fs.release();
}


void undistortImage(const cv::Mat& distortedImage, const cv::Mat& cameraMatrix, cv::Mat& newCameraMatrix, const cv::Mat& distCoeffs, cv::Mat& undistortedImage)
{
   
    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, distortedImage.size(), 0);

    cv::Mat mapx, mapy;
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, distortedImage.size(), CV_32FC1, mapx, mapy);
    cv::remap(distortedImage, undistortedImage, mapx, mapy, cv::INTER_LINEAR);
    mapx.release();
    mapy.release();
    R.release();
    return;
}

void drawMatchesRANSAC(cv::Mat& img1, std::vector<cv::KeyPoint>& keypoints1,
                       cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints2,
                       std::vector<cv::DMatch>& matches,
                       std::vector<cv::DMatch>& matchesRANSAC) {
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    
    cv::Mat imgMatchesRANSAC;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matchesRANSAC, imgMatchesRANSAC);

    cv::namedWindow("Matches");
    cv::namedWindow("Matches after RANSAC");
    
    cv::imshow("Matches", imgMatches);
    cv::imshow("Matches after RANSAC", imgMatchesRANSAC);
    cv::imwrite("./Matches.png", imgMatches);
    cv::imwrite("./MatchesRANSAC.png", imgMatchesRANSAC);
    cv::waitKey(0);
    cv::destroyAllWindows();
}



int main(int argc, char const *argv[])
{
    /* code */
    camera_param param;
    // readAndInverse("/home/xreal/xreal/two_image_pose_estimation/1403637188088318976");
    ReadParameter(param, "/home/xreal/xreal/two_image_pose_estimation/sensor.yaml");
    string image_path("/home/xreal/xreal/two_image_pose_estimation/");
    cv::Mat image_1 = cv::imread(image_path + "1403637188088318976.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_2 = cv::imread(image_path + "1403637189138319104.png", cv::IMREAD_GRAYSCALE);

    cv::Mat distCoeffs(4, 1, CV_32F);
    
    for (int i = 0; i < 4; i++) {
        distCoeffs.at<float>(i) = param.distortionCoefficients[i];
    }
    cv::Mat CameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
    CameraMatrix.at<float>(0, 0) = param.intrinsics[0];
    CameraMatrix.at<float>(1, 1) = param.intrinsics[1];
    CameraMatrix.at<float>(0, 2) = param.intrinsics[2];
    CameraMatrix.at<float>(1, 2) = param.intrinsics[3];
    CameraMatrix.at<float>(2, 2) = 1;
    cv::Mat undistortedImage1;
    cv::Mat undistortedImage2;
    cv::Mat newCameraMatrix;
    undistortImage(image_1, CameraMatrix, newCameraMatrix, distCoeffs, undistortedImage1);
    undistortImage(image_2, CameraMatrix, newCameraMatrix, distCoeffs, undistortedImage2);
    // 扣出了中间区域
    // cv::undistort(image, undistortedImage1, CameraMatrix, distCoeffs);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches, matchesRANSAC;

    // cv::undistortPoints();  
    // cv::undistortPoints();
    string path = "/home/xreal/xreal/two_image_pose_estimation/";
    
    int featureCount = 1000;  // 特征点数量
    double ransacThreshold = 1.5;  // RANSAC阈值

    featureMatchingRANSAC(undistortedImage1, undistortedImage2, keypoints1, keypoints2, matches, matchesRANSAC, featureCount, ransacThreshold);

    // drawMatchesRANSAC(undistortedImage1, keypoints1, undistortedImage2, keypoints2, matches, matchesRANSAC);

    solvePose(keypoints1, keypoints2, matches, newCameraMatrix);
    // cv::imwrite(path + "1403637188088318976_undistortedImage.jpg", undistortedImage);
    // cv::imwrite(path + "1403637188088318976_undistortedImage1.jpg", undistortedImage1);
    // cv::imshow("undistortedImage", undistortedImage1);
    // cv::imshow("undistortedImage1", undistortedImage2);
    // cv::waitKey(0);

    return 0;
}