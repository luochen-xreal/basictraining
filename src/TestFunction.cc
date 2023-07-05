#include <include/TestFunction.h>
#include <random>
#include <Eigen/Dense>

void add_print(int a, int b){
    std::cout << a + b << std::endl;
    return;
}


void solvePose(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::DMatch>& matches, 
                const cv::Mat CameraMatrix){
    
    if(matches.size() < 8){
        std::cout << "can't solve pose." << std::endl;
    }
    int n = matches.size(); // 生成1到n之间的随机数
    int k = 8;  // 随机选择的数的数量

    
    std::random_device rd;
    std::mt19937 gen(rd());

    // 使用当前时间作为种子
    std::time_t seed = std::time(nullptr);
    gen.seed(seed);

    std::uniform_int_distribution<> dis(0, n - 1);
    std::vector<int> numbers;
    for (int i = 0; i < k; ++i) {
        int num = dis(gen);
        numbers.push_back(num);
    }
    Eigen::MatrixXf A(8, 9);
    Eigen::MatrixXf E(3, 3);
    Eigen::Matrix3f K;
    K << CameraMatrix.at<float>(0, 0), CameraMatrix.at<float>(0, 1), CameraMatrix.at<float>(0, 2), 
        CameraMatrix.at<float>(1, 0), CameraMatrix.at<float>(1, 1), CameraMatrix.at<float>(1, 2), 
        CameraMatrix.at<float>(2, 0), CameraMatrix.at<float>(2, 1), CameraMatrix.at<float>(2, 2);
    Eigen::Matrix3f K_inv = K.inverse();
    for (int i = 0; i < k; ++i) {
        cv::Point2f kp1 = keypoints1[matches[numbers[i]].queryIdx].pt;
        cv::Point2f kp2 = keypoints1[matches[numbers[i]].trainIdx].pt;
        Eigen::Vector3f p1, p2;
        p1 << kp1.x, kp1.y, 1;
        p2 << kp2.x, kp2.y, 1;
        p1 = K_inv * p1;
        p2 = K_inv * p2;
        float u1 = p1.x(), v1 = p1.y(), u2 = p2.x(), v2 = p2.y(); 
        A.row(i) << u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);

    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // 提取U、S和V
    Eigen::Matrix<double, 8, 8> U = svd.matrixU();
    Eigen::Matrix<double, 9, 9> V = svd.matrixV();
    Eigen::Matrix<double, 9, 1> singularValues = svd.singularValues();
    Eigen::Matrix<double, 8, 9> diag;
    diag = diag.Zero();
    
    // 构造本质矩阵E
    Eigen::Matrix3d E = U * diag * V.transpose();

    // 使用SVD分解得到的U和V计算旋转和平移矩阵
    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;

    Eigen::Matrix3d R = U * W * V.transpose();
    Eigen::Vector3d t = U.col(8);

}