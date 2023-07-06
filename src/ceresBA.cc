#include <ceres/ceres.h>
#include "include/ceresBA.h"


void optimization(vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& R0, cv::Mat& t0, cv::Mat& R1, cv::Mat& t1){


    ceres::Problem problem;
    int n = 10;
    double para_pose[n][6];
    double para_point[500][3];
    
    ceres::Problem problem;
    for (int i = 0; i < n; ++i) {
        ceres::CostFunction* cost_function = BAReprojectionError::Create(1, 1);
        problem.AddResidualBlock(cost_function,
                                nullptr /* squared loss */,
                                para_pose[i],
                                para_point[i + 100]);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}