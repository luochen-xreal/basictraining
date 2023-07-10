#include <include/utils.h>


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






cv::Point2f pixel2cam(const cv::Point2f& p,const cv::Mat& K){
    return cv::Point2f
    (
      (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
      (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
    );
}
Eigen::Vector3f pixel2eigen(const cv::Point2f& p,const cv::Mat& K){
    return Eigen::Vector3f
    (
      (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
      (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1), 
      1
    );
}      
cv::Point2f point2pixel(cv::Mat point, cv::Mat K, cv::Mat R, cv::Mat t){
    cv::Mat p = R * point + t;
    p = p / p.at<float>(2);
    p = K * p;
    // cout << "p :" << p << endl;
    return cv::Point2f(p.at<float>(0), p.at<float>(1));
}
void points2pixelVector(vector<Eigen::Vector3f>& points, vector<cv::Point2f>& campoint,const cv::Mat K, const cv::Mat R, const cv::Mat t){
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

void reprojectionErrorStatistics(std::vector<cv::Point2f>& uv1, std::vector<cv::Point2f>& uv2){
    if (uv1.size() != uv2.size()) {
        std::cout << "Error: Input vectors have different sizes." << std::endl;
        return;
    }

    size_t numPoints = uv1.size();
    std::vector<cv::Point2f> errors(numPoints);
   
    // 计算误差向量
    for (size_t i = 0; i < numPoints; ++i) {
        errors[i] = uv2[i] - uv1[i];
    }

    // 计算误差均值
    cv::Point2f mean(0, 0);
    for (const cv::Point2f& error : errors) {
        mean += error;
    }
    mean = mean / static_cast<float>(numPoints);

    // 计算误差方差和标准差
    cv::Point2f variance(0, 0);
    for (const cv::Point2f& error : errors) {
        variance.x += std::pow(error.x - mean.x, 2);
        variance.y += std::pow(error.y - mean.y, 2);
    }
    variance /= static_cast<float>(numPoints);
    cv::Point2f standardDeviation(std::sqrt(variance.x), std::sqrt(variance.y));

    // 输出统计值
    std::cout << "Reprojection Error Statistics:" << std::endl;
    std::cout << "Mean: (" << mean.x << ", " << mean.y << ")" << std::endl;
    std::cout << "Variance: (" << variance.x << ", " << variance.y << ")" << std::endl;
    std::cout << "Standard Deviation: (" << standardDeviation.x << ", " << standardDeviation.y << ")" << std::endl;


    std::string filename = "data.txt";
    saveDataToFile(errors, filename);
}

void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


Eigen::MatrixXf cvMatToEigen(const cv::Mat& cvMat)
{
    Eigen::MatrixXf eigenMat(cvMat.rows, cvMat.cols);

    // 将 cv::Mat 的数据复制到 Eigen::MatrixXf 中
    for (int i = 0; i < cvMat.rows; ++i)
    {
        for (int j = 0; j < cvMat.cols; ++j)
        {
            eigenMat(i, j) = cvMat.at<float>(i, j);
        }
    }

    return eigenMat;
}

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

void saveDataToFile(const std::vector<cv::Point2f>& data, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cout << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (const cv::Point2f& point : data) {
        file << point.x << " " << point.y << std::endl;
    }

    file.close();
    std::cout << "Data saved to file: " << filename << std::endl;
}

void drawX(cv::Mat& image, const cv::Point& point, int size, const cv::Scalar& color, int thickness) {
    int halfSize = size / 2;

    // 绘制第一条线
    cv::line(image, cv::Point(point.x - halfSize, point.y - halfSize),
             cv::Point(point.x + halfSize, point.y + halfSize), color, thickness);

    // 绘制第二条线
    cv::line(image, cv::Point(point.x - halfSize, point.y + halfSize),
             cv::Point(point.x + halfSize, point.y - halfSize), color, thickness);
}

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