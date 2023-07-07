#include <include/utils.h>
#include <random>
#include "opencv2/viz.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <algorithm>

using namespace std;
using namespace cv;
// using namespace Eigen;
void add_print(int a, int b){
    std::cout << a + b << std::endl;
    return;
}

void eightPointMethod(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::DMatch>& matches, const cv::Mat CameraMatrix){
    
}
void solvePoseOpenCV(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier,
                std::vector<cv::DMatch>& matches, 
                const cv::Mat& CameraMatrix, cv::Mat& R, cv::Mat& t, cv::Mat& essential_matrix){
    
    vector<Point2f> points1;
    vector<Point2f> points2;
    
    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints2[matches[i].trainIdx].pt );
    }
    //-- 计算本质矩阵
    cv::Mat mask;
    essential_matrix = findEssentialMat ( points1, points2, CameraMatrix, 8, 0.9989999999999999991, 1.0, mask);
    cout<<"essential_matrix is "<< endl << essential_matrix << endl;
    int inlier = recoverPose ( essential_matrix, points1, points2, CameraMatrix, R, t, mask);
    cout << "mask size = " << mask.rows << endl;
    for(size_t i = 0; i < matches.size(); i++){
        // cout << mask.at<uchar>(i, 0) << endl;
        if(mask.at<uchar>(i, 0) == 1.0){
            
            keypoints1_inlier.push_back(keypoints1[matches[i].queryIdx]);
            keypoints2_inlier.push_back(keypoints2[matches[i].trainIdx]);
        }
    }
    // cout << "mask = " << endl << mask << endl;
    cout << "points size = " << points1.size() << endl;
    cout << "keypoints1_inlier size = " << keypoints1_inlier.size() << endl;
    cout << "inlier size = " << inlier << endl;
    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    
}
void solvePose(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat CameraMatrix){
    
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints2[matches[i].trainIdx].pt );
    }
    //-- 计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, CameraMatrix);
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    // Mat homography_matrix;
    // homography_matrix = findHomography ( points1, points2, CV_RANSAC, 3 );
    // cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    Mat R,t;
    recoverPose ( essential_matrix, points1, points2, CameraMatrix, R, t);
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;

    //
    if(matches.size() < 8){
        std::cout << "can't solve pose." << std::endl;
    }

    //
    
    int k = 8;  // 随机选择的数的数量

    
    std::random_device rd;
    std::mt19937 gen(rd());
    const int kMaxIterations = 7;
    // 使用当前时间作为种子
    std::time_t seed = std::time(nullptr);
    gen.seed(seed);
    for(size_t i = 0; i < kMaxIterations; i++){

    }
    int n = matches.size(); // 生成1到n之间的随机数
    std::uniform_int_distribution<> dis(0, n - 1);
    std::vector<int> numbers;
    for (int i = 0; i < k; ++i) {
        int num = dis(gen);
        if(std::find(numbers.begin(), numbers.end(), num) != numbers.end()){
            i--;
            continue;
        }
        numbers.push_back(num);
        // std::cout << "num:" << num << std::endl;
    }
    // std::cout << "matches.size = " << matches.size() << std::endl;
    // std::cout << "numbers.size = " << numbers.size() << std::endl;
    Eigen::MatrixXf A(k, 9);
    Eigen::MatrixXf E(3, 3);
    Eigen::Matrix3f K;
    // std::cout << "CameraMatrix.size = " << CameraMatrix.rows << "," << CameraMatrix.cols << std::endl;
    K << CameraMatrix.at<float>(0, 0), CameraMatrix.at<float>(0, 1), CameraMatrix.at<float>(0, 2), 
        CameraMatrix.at<float>(1, 0), CameraMatrix.at<float>(1, 1), CameraMatrix.at<float>(1, 2), 
        CameraMatrix.at<float>(2, 0), CameraMatrix.at<float>(2, 1), CameraMatrix.at<float>(2, 2);
    Eigen::Matrix3f K_inv = K.inverse();
    // std::cout << "K= " << std::endl << K << std::endl;
    // std::cout << "K_inv= " << std::endl << K_inv << std::endl;
    for (int i = 0; i < k; ++i) {
        cv::Point2f kp1 = keypoints1[matches[numbers[i]].queryIdx].pt;
        cv::Point2f kp2 = keypoints1[matches[numbers[i]].trainIdx].pt;
        Eigen::Vector3f p1, p2;
        p1 << kp1.x, kp1.y, 1;
        p2 << kp2.x, kp2.y, 1;
        p1 = K_inv * p1;
        p2 = K_inv * p2;
        // std::cout << "p1=" << p1 << std::endl;

        // std::cout << "p2=" << p2 << std::endl;
        float u1 = p1.x(), v1 = p1.y(), u2 = p2.x(), v2 = p2.y(); 
        A.row(i) << u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1;
    }



    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf V = svd.matrixV();
    Eigen::VectorXf e = V.col(svd.singularValues().size() - 1);
    // e = e / e(8);
    std::cout << "svd.singularValues().size() = " << svd.singularValues().size() << std::endl;
    E.row(0) << e(0), e(1), e(2);
    E.row(1) << e(3), e(4), e(5);
    E.row(2) << e(6), e(7), e(8);


    cout << "essential_matrix is " << endl;
    cout << E << endl;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd2(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    float angle = 90.0 * M_PI / 180.0;
    // 创建旋转矩阵
    Eigen::Matrix3f Z;
    Z = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ());
    // std::cout << "Z = " << std::endl << Z << std::endl;
    Eigen::Matrix3f Sigma = svd2.singularValues().asDiagonal();
    Sigma(2, 2) = 0;
    Sigma(0, 0) = 1;
    Sigma(1, 1) = 1;
    U = svd2.matrixU();
    V = svd2.matrixV();

    Eigen::Matrix3f R1 = U * Z * V.transpose();
    // std::cout << "|R1| = " << R1.determinant() << std::endl;
    Eigen::Matrix3f R2 = U * Z.transpose() * V.transpose();
    if(R1.determinant() < 0) R1 = -R1;
    if(R2.determinant() < 0) R2 = -R2;
    // checkRT()
    Eigen::Vector3f t1 = U.col(2);
    t1 = t1 / t1.norm();
    Eigen::Vector3f t2 = -t1;
    // 判断哪一组符合要求
    // std::cout << "R1=" << std::endl << R1 << std::endl;
    // std::cout << "R2=" << std::endl << R2 << std::endl;
    // std::cout << "t1=" << std::endl << t1 << std::endl;
    // std::cout << "t2=" << std::endl << t2 << std::endl;
    Eigen::Matrix3f t_x;
    t_x << 0, -t1(2), t1(1),t1(2),0, -t1(0),-t1(1),t1(0),0;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Point2f kp1 = keypoints1[matches[i].queryIdx].pt;
        cv::Point2f kp2 = keypoints1[matches[i].trainIdx].pt;
        Eigen::Vector3f p1, p2;
        p1 << kp1.x, kp1.y, 1;
        p2 << kp2.x, kp2.y, 1;
        p1 = K_inv * p1;
        p2 = K_inv * p2;
        // std::cout << "p1=" << p1 << std::endl;

        // std::cout << (p2.transpose() * t_x * R1 * p1) << std::endl;
    }














    // viz::Viz3d myWindow("Coordinate Frame"); 
    //----------------------读入相机位姿----------------------
    // Vec3d cam1_pose(0, 0, 0), cam1_focalPoint(0, 0, 1), cam1_y_dir(0, -1, 0); // 设置相机的朝向（光轴朝向）
    // Affine3d cam_3_pose = viz::makeCameraPose(cam1_pose, cam1_focalPoint, cam1_y_dir); // 设置相机位置与朝向
    
    // myWindow.showWidget("World_coordinate",viz::WCoordinateSystem(),cam_3_pose); // 创建3号相机位于世界坐标系的原点
    // 创建R\T
    Matx33d PoseR_0,PoseR_1,PoseR_2; // 旋转矩阵
    Vec3d PoseT_0,PoseT_1,PoseT_2; // 平移向量
    PoseR_0 = Matx33d(R1(0, 0),R1(0, 1),R1(0, 2),R1(1, 0),R1(1, 1),R1(1, 2),R1(2, 0),R1(2, 1),R1(2, 2));
    PoseT_0 = Vec3d(t1(0), t1(1), t1(2));
    PoseR_1 = Matx33d(R2(0, 0),R2(0, 1),R2(0, 2),R2(1, 0),R2(1, 1),R2(1, 2),R2(2, 0),R2(2, 1),R2(2, 2));
    PoseT_1 = Vec3d(t2(0), t2(1), t2(2));
    // PoseR_1 = Matx33d(-0.821903,0.0251458,-0.569073,  -0.56962,-0.0416208,0.820854,-0.00304428,0.998817,0.0485318);
    // PoseT_1 = Vec3d(0.0754863,-0.108494,  0.113143 ) * 10;
    // PoseR_2 = Matx33d(0.880609,0.0740675,-0.468019,-0.469291,-0.000261475,-0.883044, -0.0655272,0.997253,0.034529);
    // PoseT_2 = Vec3d(0.0624015,0.109845,0.119439) * 10;
    
    Affine3d Transpose03(PoseR_0,PoseT_0); // 03相机变换矩阵
    Affine3d Transpose13(PoseR_1,PoseT_1); // 13相机变换矩阵
    // Affine3d Transpose23(PoseR_2,PoseT_2); // 23相机变换矩阵
    // ----------------------设置坐标系----------------------
    // myWindow.showWidget("Cam0",viz::WCoordinateSystem(),Transpose03);
    // myWindow.showWidget("Cam1",viz::WCoordinateSystem(),Transpose13);
    // myWindow.showWidget("Cam2",viz::WCoordinateSystem(),Transpose23);
    // ----------------------显示----------------------
    // myWindow.spin();

}


void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,  
                const cv::Mat CameraMatrix, const cv::Mat& R, const cv::Mat& t, vector<Eigen::Vector3d> &points){
    cv::Mat T1 = (cv::Mat_<float>(3, 4) <<  1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<  R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), t.at<float>(0, 0),
                                            R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), t.at<float>(1, 0),
                                            R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), t.at<float>(2, 0));
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for ( int i = 0; i < ( int ) keypoints1.size(); i++ )
    {
        points1.push_back ( pixel2cam(keypoints1[i].pt, CameraMatrix) );
        points2.push_back ( pixel2cam(keypoints2[i].pt, CameraMatrix) );

    }
    // cout << R.type() << endl;
    // cout << "T1" << T1 << endl;
    // cout << "T2" << T2 << endl;
    // for ( int i = 0; i < ( int ) matches.size(); i++ )
    // {
    //     cout << points1[i] << endl;
    //     cout << points2[i] << endl;
    // }
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, points1, points2, pts_4d);
    vector<cv::Point3f> points_3d;
    for(int i = 0; i < pts_4d.cols; i++){
        Mat x = pts_4d.col(i);
        
        x /= x.at<float>(3, 0); // 归一化
        Eigen::Vector3d p(x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        points.push_back(p);
        // cout << "p:" << p << endl;
    }
}


cv::Point2f pixel2cam(const cv::Point2f& p,const cv::Mat& K){
    return cv::Point2f
    (
      (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
      (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
    );
}   
cv::Point2f point2pixel(cv::Mat point, cv::Mat K, cv::Mat R, cv::Mat t){
    cv::Mat p = R * point + t;
    p = p / p.at<float>(2);
    p = K * p;
    // cout << "p :" << p << endl;
    return cv::Point2f(p.at<float>(0) / p.at<float>(2), p.at<float>(1) / p.at<float>(2));
}
void points2pixelVector(vector<Eigen::Vector3d>& points, vector<cv::Point2f>& campoint,const cv::Mat K, const cv::Mat R, const cv::Mat t){
    campoint.reserve(points.size());

    for(size_t i = 0; i < points.size(); i++){
        cv::Mat p = cv::Mat(3, 1, CV_32F);
        p.at<float>(0) = points[i].x();
        p.at<float>(1) = points[i].y();
        p.at<float>(2) = points[i].z();
        // cout << "p :" << p << endl;
        campoint.push_back(point2pixel(p, K, R, t));
    }

}