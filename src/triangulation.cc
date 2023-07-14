#include "include/triangulation.h"
using namespace std;

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,  
                const cv::Mat CameraMatrix, const cv::Mat& R, const cv::Mat& t, vector<Eigen::Vector3f> &points){
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
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, points1, points2, pts_4d);
    vector<cv::Point3f> points_3d;
    for(int i = 0; i < pts_4d.cols; i++){
        cv::Mat x = pts_4d.col(i);
        
        x /= x.at<float>(3, 0); // 归一化
        Eigen::Vector3f p(x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        points.push_back(p);
        
    }
}

void triangulation(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2,  
                const cv::Mat CameraMatrix, const Eigen::Matrix3f& R, const Eigen::Vector3f& t, vector<Eigen::Vector3f> &points){
    vector<Eigen::Vector3f> points1;
    vector<Eigen::Vector3f> points2;

    for ( int i = 0; i < ( int ) keypoints1.size(); i++ )
    {

        points1.push_back ( pixel2eigen(keypoints1[i].pt, CameraMatrix) );
        points2.push_back ( pixel2eigen(keypoints2[i].pt, CameraMatrix) );

    }
    // cout << R.type() << endl;
    // cout << "T1" << T1 << endl;
    // cout << "T2" << T2 << endl;
    // for ( int i = 0; i < ( int ) matches.size(); i++ )
    // {
    //     cout << points1[i] << endl;
    //     cout << points2[i] << endl;
    // }
    Eigen::Vector4f pts_4d;
    Eigen::Matrix<float, 3, 4> P0 = Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::Matrix<float, 3, 4> P1;
    P1.block<3, 3>(0, 0) = R;
    P1.col(3) = t;
    
    vector<cv::Point3f> points_3d;
    for(int i = 0; i < points1.size(); i++){
        Triangulate(P0, P1, points1[i], points2[i], pts_4d);
        Eigen::Vector4f x = pts_4d;
        
        x /= x(3); // 归一化
        Eigen::Vector3f p(x(0),
                            x(1),
                            x(2));
        points.push_back(p);
        
    }
}

void Triangulate(const Eigen::MatrixXf &P1, const Eigen::MatrixXf &P2, const Eigen::Vector3f &kp1, const Eigen::Vector3f &kp2, 
Eigen::Vector4f &x3D)
{
    
    Eigen::Matrix4f A;
    A.row(0) = kp1.x() * P1.row(2) - P1.row(0);
    A.row(1) = kp1.y() * P1.row(2) - P1.row(1);
    A.row(2) = kp2.x() * P2.row(2) - P2.row(0);
    A.row(3) = kp2.y() * P2.row(2) - P2.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4f> svd2(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix4f V = svd2.matrixV();
    x3D = V.col(3);
    x3D = x3D / x3D(3);
}   