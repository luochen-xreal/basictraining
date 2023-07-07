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
        T predicted_y = p[1] / p[2];
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return(new ceres::AutoDiffCostFunction<BAReprojectionError, 2, 6, 3>(
                    new BAReprojectionError(observed_x, observed_y)));
        
    }

    double observed_x;
    double observed_y;
};


struct ProjectionErrorFactor : public ceres::SizedCostFunction<2, 6, 3>
{
    public:
    ProjectionErrorFactor(const double x, const double y) : observed_x(x), observed_y(y){}
    virtual bool Evaluate(const double *const *parameters, double *residuals, double **jacobians);
    double observed_x;
    double observed_y;
};

void optimization(const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, vector<Eigen::Vector3d>& points, cv::Mat& R0, cv::Mat& t0, cv::Mat& R1, cv::Mat& t1);