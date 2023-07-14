#include <ceres/ceres.h>
#include "include/ceresBA.h"
#include "include/utility.h"
using namespace std;

void optimization(const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, vector<Eigen::Vector3f>& points, cv::Mat& R0, cv::Mat& t0, cv::Mat& R1, cv::Mat& t1){


    ceres::Problem problem;

    cv::Mat rvec0, rvec1;
    cv::Rodrigues(R0, rvec0);
    cv::Rodrigues(R1, rvec1);
    // cout << "t0:" << t0 << endl;
    // cout << "t1:" << t1 << endl;
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
        // cout << "para_point:" << para_point[i][0] << "," << para_point[i][1] << "," << para_point[i][2] << endl;
        problem.AddParameterBlock(para_point[i], 3);
        // cout << "point " << i << ": " << points[i].transpose() << endl;
    }
    cout << "points size = " << points.size() << endl;
    // ceres::HuberLoss* loss_function = new ceres::HuberLoss(1);
    problem.AddParameterBlock(para_pose[0], 6);
    problem.AddParameterBlock(para_pose[1], 6);
    if(0){
        for(size_t j = 0; j < n; ++j){
            ceres::CostFunction* cost_function = BAReprojectionError::Create(static_cast<double>(points1[j].x), static_cast<double>(points1[j].y));
            problem.AddResidualBlock(cost_function,
                                nullptr,
                                para_pose[0],
                                para_point[j]);
            // problem.SetParameterBlockConstant(para_point[j]);
        }

        for(size_t j = 0; j < n; ++j){
            ceres::CostFunction* cost_function = BAReprojectionError::Create(static_cast<double>(points2[j].x), static_cast<double>(points2[j].y));
            problem.AddResidualBlock(cost_function,
                                nullptr,
                                para_pose[1],
                                para_point[j]);
        }
    }else{
        for(size_t j = 0; j < n; ++j){
            ProjectionErrorFactor* cost_function = new ProjectionErrorFactor(static_cast<double>(points1[j].x), static_cast<double>(points1[j].y));
            problem.AddResidualBlock(cost_function,
                                nullptr,
                                para_pose[0],
                                para_point[j]);
            // problem.SetParameterBlockConstant(para_point[j]);
        }

        for(size_t j = 0; j < n; ++j){
            ProjectionErrorFactor* cost_function = new ProjectionErrorFactor(static_cast<double>(points2[j].x), static_cast<double>(points2[j].y));
            problem.AddResidualBlock(cost_function,
                                nullptr,
                                para_pose[1],
                                para_point[j]);
        }
    }
    problem.SetParameterBlockConstant(para_pose[0]);
  
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    // options.max_num_iterations = 40;
    // options.check_gradients = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    cout << "optimized para_pose0 = " << para_pose[0][0] << "," << para_pose[0][1] << "," << para_pose[0][2] << "," << para_pose[0][3] << "," << para_pose[0][4] << "," << para_pose[0][5] << endl;
    cout << "optimized para_pose1 = " << para_pose[1][0] << "," << para_pose[1][1] << "," << para_pose[1][2] << "," << para_pose[1][3] << "," << para_pose[1][4] << "," << para_pose[1][5] << endl;
 

    for(size_t i = 0; i < n; i++){
        points[i](0) = para_point[i][0];
        points[i](1) = para_point[i][1];
        points[i](2) = para_point[i][2];
        
    }
    cv::Mat axis0 = cv::Mat(3, 1, CV_32F);
    cv::Mat axis1 = cv::Mat(3, 1, CV_32F);

    for (int i = 0; i < 3; ++i) {
        axis0.at<float>(i, 0) = para_pose[0][i];  // 赋值
        axis1.at<float>(i, 0) = para_pose[1][i];  // 赋值
        t0.at<float>(i, 0) = para_pose[0][i + 3];
        t1.at<float>(i, 0) = para_pose[1][i + 3];
    }
   
    cv::Rodrigues(axis0, R0);
    cout << "R0 = " << endl << R0 << endl;
    cout << "t0 = " << endl << t0 << endl;
    cv::Rodrigues(axis1, R1);
    cout << "R1 = " << endl << R1 << endl;
    cout << "t1 = " << endl << t1 << endl;

    std::cout << summary.FullReport() << "\n";
    std::cout << "耗时：" << summary.total_time_in_seconds << "秒" << std::endl;

}


bool ProjectionErrorFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const{
    Eigen::Vector3d axis(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d t(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d p(parameters[1][0], parameters[1][1], parameters[1][2]);
    // 信息矩阵开更
    Eigen::Matrix2d sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
    Eigen::AngleAxisd rv(axis.norm(), axis.normalized());
    double theta = axis.norm();
    Eigen::Vector3d a = axis.normalized();
    Eigen::Matrix3d Rcw = rv.toRotationMatrix();
    // Eigen::Matrix3d Rcw = axis2Matrix(axis);
    // cout << "Rcw" << Rcw << endl;
    Eigen::Vector3d p_cam = Rcw * p + t;
    Eigen::Vector3d p_xy = p_cam / p_cam.z(); 
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual(0) = observed_x - p_xy(0);
    residual(1) = observed_y - p_xy(1); 
    residual = sqrt_info * residual;
    Eigen::Matrix<double, 2, 3> jacobPc;
    jacobPc <<  1 / p_cam(2), 0, - p_cam(0) / (p_cam(2) * p_cam(2)),
                0, 1 / p_cam(2), - p_cam(1) / (p_cam(2) * p_cam(2));
    jacobPc = - sqrt_info * jacobPc;
    Eigen::Matrix<double, 3, 3> jacobXi;
    Eigen::Matrix3d Jl;
    // cout << "theta = " << theta << endl;
    // 保证 theta不为0
    if(theta == 0) theta = 1.0 / 1e12;
    Jl = sin(theta) / theta * Eigen::Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() 
            + (1 - cos(theta)) / theta * Utility::skewSymmetric(a);
    // cout << "Jl = " << Jl << endl;
    jacobXi = - Utility::skewSymmetric(Rcw * p);
    Eigen::Matrix<double, 3, 3> jacobP = Rcw;
    if(jacobians){
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            jacobian_pose.block<2, 3>(0, 0) = jacobPc * jacobXi * Jl;
            // jacobian_pose.block<2, 3>(0, 0) = jacobPc * jacobXi;
            jacobian_pose.block<2, 3>(0, 3) = jacobPc * Eigen::Matrix<double, 3, 3>::Identity();
        }
        if(jacobians[1]){
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_point(jacobians[1]);
            jacobian_point = jacobPc * jacobP;
        }
    }
    return true;
 }



// bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
// {
//     Eigen::Map<const Eigen::Vector3d> _p(x);
//     Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

//     Eigen::Map<const Eigen::Vector3d> dp(delta);

//     Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

//     Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
//     Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

//     p = _p + dp;
//     q = (_q * dq).normalized();

//     return true;
// }
// bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
// {
//     Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
//     j.topRows<6>().setIdentity();
//     j.bottomRows<1>().setZero();

//     return true;
// }
