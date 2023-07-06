#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
struct BAReprojectionError{

    BAReprojectionError(double observed_x, double observed_y): observed_x(observed_x),observed_y(observed_y){}

    
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const{
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        T predicted_x = p[0] / p[2];
        T predicted_y = p[0] / p[2];
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(ovserved_y);
    }
    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return(new ceres::AutoDiffCostFunction<BAReprojectionError, 2, 6, 3>(
                    new BAReprojectionError(observed_x, observed_y)));
        
    }

    double observed_x;
    double observed_y;
};

void optimization(vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& R0, cv::Mat& t0, cv::Mat& R1, cv::Mat& t1);