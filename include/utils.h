#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
using namespace std;
void add_print(int a, int b);

void solvePoseOpenCV(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat& CameraMatrix, cv::Mat& R, cv::Mat& t, cv::Mat& essential_matrix);

void solvePose(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& matches, const cv::Mat CameraMatrix);

cv::Point2f pixel2cam(const cv::Point2f& point,const cv::Mat& K);

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,
                const cv::Mat CameraMatrix, const cv::Mat& R, const cv::Mat& t, std::vector<Eigen::Vector3d> &points);


void drawReprojection(cv::Mat& img, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2){
    cv::Scalar color1(255, 0, 0);  // 连线和点的颜色
    cv::Mat colorImage;
    cv::cvtColor(img, colorImage, cv::COLOR_GRAY2BGR); 
    // 绘制第一组对应点
    cout << "points1.size = " << points1.size() << endl;
    for (const auto& point : points1) {
        cv::circle(colorImage, point, 2, color1, cv::MARKER_CROSS);  // 绘制点
    }
    cv::Scalar color2(0, 0, 255);  // 连线和点的颜色
    // 绘制第二组对应点
    for (const auto& point : points2) {
        // cout << "point:" << point << endl;
        cv::circle(colorImage, point, 3, color2, cv::MARKER_CROSS);  // 绘制点
    }

    // 绘制对应点之间的连线
    cv::Scalar color3(0, 255, 0);
    for (size_t i = 0; i < points1.size(); ++i) {
        cv::line(colorImage, points1[i], points2[i], color3, 1);  // 绘制连线
    }
    double scale_factor = 2.0;  
    int new_width = static_cast<int>(colorImage.cols * scale_factor);
    int new_height = static_cast<int>(colorImage.rows * scale_factor);

    // 创建放大后的图像
    cv::Mat resized_image;
    cv::resize(colorImage, resized_image, cv::Size(new_width, new_height));

    cv::imshow("Correspondences", resized_image);
    cv::waitKey(0);
}
cv::Point2f point2pixel(cv::Mat point, cv::Mat K, cv::Mat R, cv::Mat t);

void points2pixelVector(std::vector<Eigen::Vector3d>& points, std::vector<cv::Point2f>& campoint, const cv::Mat K, const cv::Mat R, const cv::Mat t);