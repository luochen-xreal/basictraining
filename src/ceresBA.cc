#include <ceres/ceres.h>
#include "include/ceresBA.h"
using namespace std;

void optimization(const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, vector<Eigen::Vector3d>& points, cv::Mat& R0, cv::Mat& t0, cv::Mat& R1, cv::Mat& t1){


    ceres::Problem problem;

    cv::Mat rvec0, rvec1;
    cv::Rodrigues(R0, rvec0);
    cv::Rodrigues(R1, rvec1);
    cout << "t0:" << t0 << endl;
    cout << "t1:" << t1 << endl;
    // cout << "rvec0:" << rvec0 << endl;
    // cout << "rvec1:" << rvec1 << endl;
    double para_pose[2][6];
    double para_point[1000][3];
    size_t n = points.size();
    para_pose[0][0] = rvec0.at<float>(0,0);
    para_pose[0][1] = rvec0.at<float>(1,0);
    para_pose[0][2] = rvec0.at<float>(2,0);
    para_pose[0][3] = t0.at<float>(0,0);
    para_pose[0][4] = t0.at<float>(1,0);
    para_pose[0][5] = t0.at<float>(2,0);  
    para_pose[1][0] = rvec1.at<float>(0,0);
    para_pose[1][1] = rvec1.at<float>(1,0);
    para_pose[1][2] = rvec1.at<float>(2,0);
    para_pose[1][3] = t1.at<float>(0,0);
    para_pose[1][4] = t1.at<float>(1,0);
    para_pose[1][5] = t1.at<float>(2,0);  
    cout << "para_pose0 = " << para_pose[0][0] << "," << para_pose[0][1] << "," << para_pose[0][2] << "," << para_pose[0][3] << "," << para_pose[0][4] << "," << para_pose[0][5] << endl;
    cout << "para_pose1 = " << para_pose[1][0] << "," << para_pose[1][1] << "," << para_pose[1][2] << "," << para_pose[1][3] << "," << para_pose[1][4] << "," << para_pose[1][5] << endl;
    for(size_t i = 0; i < points.size(); i++){
        para_point[i][0] = points[i].x();
        para_point[i][1] = points[i].y();
        para_point[i][2] = points[i].z();
        // cout << "point " << i << ": " << points[i].transpose() << endl;
    }
    cout << "points size = " << points.size() << endl;
    // ceres::HuberLoss* loss_function = new ceres::HuberLoss(1);
    for(size_t j = 0; j < n; ++j){
        ceres::CostFunction* cost_function = BAReprojectionError::Create(static_cast<double>(points1[j].x), static_cast<double>(points1[j].y));
        problem.AddResidualBlock(cost_function,
                            nullptr,
                            para_pose[0],
                            para_point[j]);
        problem.SetParameterBlockConstant(para_point[j]);
    }

    for(size_t j = 0; j < n; ++j){
        ceres::CostFunction* cost_function = BAReprojectionError::Create(static_cast<double>(points2[j].x), static_cast<double>(points2[j].y));
        problem.AddResidualBlock(cost_function,
                            nullptr,
                            para_pose[1],
                            para_point[j]);
    }
    problem.SetParameterBlockConstant(para_pose[0]);
  
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << "optimized para_pose0 = " << para_pose[0][0] << "," << para_pose[0][1] << "," << para_pose[0][2] << "," << para_pose[0][3] << "," << para_pose[0][4] << "," << para_pose[0][5] << endl;
    cout << "optimized para_pose1 = " << para_pose[1][0] << "," << para_pose[1][1] << "," << para_pose[1][2] << "," << para_pose[1][3] << "," << para_pose[1][4] << "," << para_pose[1][5] << endl;
    // double total_error = 0.0;
    // std::vector<double> residuals;
    // // problem.Evaluate(ceres::Problem::EvaluateOptions(), &total_error, &residuals, nullptr, nullptr);

    // // 输出优化后的误差
    // std::cout << "优化后的总误差: " << total_error << std::endl;

    // 输出每个残差项的误差
    // std::cout << "每个残差项的误差:" << std::endl;
    // for (int i = 0; i < residuals.size(); ++i) {
    //     std::cout << "残差项 " << i << ": " << residuals[i] << std::endl;
    // }

    for(size_t i = 0; i < n; i++){
        points[i](0) = para_point[i][0];
        points[i](1) = para_point[i][1];
        points[i](2) = para_point[i][2];
        // cout << "point " << i << ": " << points[i].transpose() << endl;
    }
    cv::Mat axis0 = cv::Mat(3, 1, CV_32F);
    cv::Mat axis1 = cv::Mat(3, 1, CV_32F);

    for (int i = 0; i < 3; ++i) {
        axis0.at<float>(i, 0) = para_pose[0][i];  // 赋值
        axis1.at<float>(i, 0) = para_pose[1][i];  // 赋值
        t0.at<float>(i, 0) = para_pose[0][i + 3];
        t1.at<float>(i, 0) = para_pose[1][i + 3];
    }
    cout << "axis1" << axis1 << endl;
    cv::Rodrigues(axis0, R0);
    cout << "R0" << R0 << endl;
    cout << "t0" << t0 << endl;
    cv::Rodrigues(axis1, R1);
    cout << "R1" << R1 << endl;
    cout << "t1" << t1 << endl;
    std::cout << summary.FullReport() << "\n";
}
// R is 
// [0.9921183855150789, 0.09141769146790107, -0.08569664409784682;
//  -0.08996328963516888, 0.9957301365048256, 0.02069062043777967;
//  0.08722221988101465, -0.01281799293029853, 0.9961064116931818]
// t is 
// [-0.7733309321805336;
//  0.1849750159947557;
//  -0.6064185953535122]
// t0:[0;
//  0;
//  0]
// t1:[-0.7733309321805336;
//  0.1849750159947557;
//  -0.6064185953535122]


 bool ProjectionErrorFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians){
    Eigen::Vector3d axis(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d t(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d p(parameters[0][0], parameters[0][1], parameters[0][2]);
    // 创建AngleAxis对象
    Eigen::AngleAxisd rv(axis.norm(), axis);
    Eigen::Matrix3d R = rv.toRotationMatrix();
// 将AngleAxis转换为旋转矩阵
    Eigen::Vector3d p_cam = R * p + t;
    p_cam = p_cam / p_cam.z(); 
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual(0) = p_cam(0) - observed_x;
    residual(1) = p_cam(1) - observed_y; 


    if(jacobians){
        if(jacobians[0]){

        }
        if(jacobians[1]){
            
        }
    }
 }