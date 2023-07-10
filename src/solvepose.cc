#include "include/solvepose.h"

void solvePoseOpenCV(std::vector<cv::KeyPoint>& keypoints1, 
                std::vector<cv::KeyPoint>& keypoints2, 
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier,
                std::vector<cv::DMatch>& matches, 
                const cv::Mat& CameraMatrix, cv::Mat& R, cv::Mat& t, cv::Mat& essential_matrix){
    
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;
    
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
        // keypoints1_inlier.push_back(keypoints1[matches[i].queryIdx]);
        // keypoints2_inlier.push_back(keypoints2[matches[i].trainIdx]);
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
                std::vector<cv::KeyPoint>& keypoints1_inlier,
                std::vector<cv::KeyPoint>& keypoints2_inlier,
                std::vector<cv::DMatch>& matches, 
                const cv::Mat CameraMatrix, Eigen::Matrix3f& R_,
                Eigen::Vector3f& t_, Eigen::Matrix3f& essential_matrix){
    
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints2[matches[i].trainIdx].pt );
        keypoints1_inlier.push_back(keypoints1[matches[i].queryIdx]);
        keypoints2_inlier.push_back(keypoints2[matches[i].trainIdx]);
        // cout << "points1" << points1[i] << endl;
        // cout << "points2" << points2[i] << endl;
    }
    //-- opencv计算本质矩阵，做参考
    
    // essential_matrix = findEssentialMat ( points1, points2, CameraMatrix);
    
    // //-- 从本质矩阵中恢复旋转和平移信息.
    
    // recoverPose ( essential_matrix, points1, points2, CameraMatrix, R, t);
    // cout<<"R is "<<endl<<R<<endl;
    // cout<<"t is "<<endl<<t<<endl;

    
    if(matches.size() < 8){
        std::cout << "can't solve pose." << std::endl;
    }

    //
    std::vector<cv::Point2f> vNormalizedPoints1;
    std::vector<cv::Point2f> vNormalizedPoints2;
    cv::Mat T1, T2;
    //keypoints1
    Normalize(keypoints1, vNormalizedPoints1, T1);
    Normalize(keypoints2, vNormalizedPoints2, T2);
    Eigen::MatrixXf T1_ = cvMatToEigen(T1);
    Eigen::MatrixXf T2_ = cvMatToEigen(T2);
    // cout << "T1" << T1_ << endl;
    // cout << "T2" << T2_ << endl;
    int k = 8; 
    std::random_device rd;
    std::mt19937 gen(rd());
    const int kMaxIterations = 7;
    // 使用当前时间作为种子
    std::time_t seed = std::time(nullptr);
    gen.seed(seed);
    for(size_t i = 0; i < kMaxIterations; i++){

    }
    int n = matches.size(); // 生成1到n之间的随机数
    cout << "n = " << n << endl;
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
    Eigen::MatrixXf F(3, 3);
    Eigen::Matrix3f K;
    K << CameraMatrix.at<float>(0, 0), CameraMatrix.at<float>(0, 1), CameraMatrix.at<float>(0, 2), 
        CameraMatrix.at<float>(1, 0), CameraMatrix.at<float>(1, 1), CameraMatrix.at<float>(1, 2), 
        CameraMatrix.at<float>(2, 0), CameraMatrix.at<float>(2, 1), CameraMatrix.at<float>(2, 2);
    Eigen::Matrix3f K_inv = K.inverse();

    for (int i = 0; i < k; ++i) {
        cv::Point2f kp1 = vNormalizedPoints1[matches[numbers[i]].queryIdx];
        cv::Point2f kp2 = vNormalizedPoints2[matches[numbers[i]].trainIdx];
        Eigen::Vector3f p1, p2;
        p1 << kp1.x, kp1.y, 1;
        p2 << kp2.x, kp2.y, 1;
        float u1 = p1.x(), v1 = p1.y(), u2 = p2.x(), v2 = p2.y(); 
        A.row(i) << u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1;
    }



    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf V = svd.matrixV();
    Eigen::VectorXf f = V.col(8);
    // e = e / e(8);
    F.row(0) << f(0), f(1), f(2);
    F.row(1) << f(3), f(4), f(5);
    F.row(2) << f(6), f(7), f(8);
    F = T2_.transpose() * F * T1_;
    cout << "E is " << endl;
    cout << K.transpose() * F * K << endl;
    Eigen::Matrix3f E = K.transpose() * F * K;
    std::vector<Eigen::Vector3f> unpoints1, unpoints2;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Point2f kp1 = keypoints1[matches[i].queryIdx].pt;
        cv::Point2f kp2 = keypoints1[matches[i].trainIdx].pt;
        Eigen::Vector3f p1, p2;
        p1 << kp1.x, kp1.y, 1;
        p2 << kp2.x, kp2.y, 1;
        p1 = K_inv * p1;
        p2 = K_inv * p2;
        unpoints1.push_back(p1);
        unpoints2.push_back(p2);
        // std::cout << "p1=" << p1 << std::endl;
        // std::cout << (p2.transpose() * t_x * R1 * p1) << std::endl;
    }
    
    int inlier = recoverPose ( E, unpoints1, unpoints2, K, R_, t_);

    cout << "inlier = " << inlier << endl;
    cout << "R_ = " << endl << R_ << endl;
    cout << "t_ = " << endl << t_ << endl;


}


int recoverPose(const Eigen::Matrix3f E,const std::vector<Eigen::Vector3f>& _points1,const std::vector<Eigen::Vector3f>& _points2, const Eigen::Matrix3f& _cameraMatrix,
                    Eigen::Matrix3f& _R,
                    Eigen::Vector3f& _t)
{

    Eigen::JacobiSVD<Eigen::Matrix3f> svd2(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // 创建旋转矩阵
    Eigen::Matrix3f Z;
    Z <<    0, -1, 0, 
            1, 0, 0,
            0, 0, 1;
    Eigen::Matrix3f Sigma = svd2.singularValues().asDiagonal();
    Sigma(2, 2) = 0;
    Sigma(0, 0) = 1;
    Sigma(1, 1) = 1;
    auto U = svd2.matrixU();
    auto V = svd2.matrixV();

    Eigen::Matrix3f R1 = U * Z * V.transpose();
    Eigen::Matrix3f R2 = U * Z.transpose() * V.transpose();
    if(R1.determinant() < 0) R1 = -R1;
    if(R2.determinant() < 0) R2 = -R2;
    // checkRT()
    Eigen::Vector3f t = U.col(2);
    t = t / t.norm();
    size_t n = _points1.size();
    
    Eigen::Matrix3f cameraMatrix = _cameraMatrix.cast<float>();
    // int npoints = points1.cols();
    // assert(npoints >= 0 && points2.cols() == npoints);
    assert(cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3);
    double fx = cameraMatrix(0, 0);
    double fy = cameraMatrix(1, 1);
    double cx = cameraMatrix(0, 2);
    double cy = cameraMatrix(1, 2);
    Eigen::Matrix<float, 3, 4> P0 = Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::Matrix<float, 3, 4> P1, P2, P3, P4;
    P1.block<3, 3>(0, 0) = R1;
    P1.col(3) = t;
    P2.block<3, 3>(0, 0) = R2;
    P2.col(3) = t;
    P3.block<3, 3>(0, 0) = R1;
    P3.col(3) = -t;
    P4.block<3, 3>(0, 0) = R2;
    P4.col(3) = -t;
    int good1 = 0;
    int good2 = 0;
    int good3 = 0;
    int good4 = 0;
    for(size_t i = 0; i < n; i++){
        auto points1 = _points1[i];
        auto points2 = _points2[i];
        points1.row(0) = (points1.row(0).array() - cx) / fx;
        points2.row(0) = (points2.row(0).array() - cx) / fx;
        points1.row(1) = (points1.row(1).array() - cy) / fy;
        points2.row(1) = (points2.row(1).array() - cy) / fy;
        double dist = 50.0;
        Eigen::Vector4f Q;
        Triangulate(P0, P1, points1, points2, Q);
        bool mask1 = (Q(2, 0) * Q(3, 0)) > 0;
        Q(0, 0) /= Q(3, 0);
        Q(1, 0) /= Q(3, 0);
        Q(2, 0) /= Q(3, 0);
        Q(3, 0) /= Q(3, 0);
        mask1 = ((Q(2, 0) < dist) && mask1);
        Eigen::Vector3f Q_ = P1 * Q;
        mask1 = (Q_(2, 0) > 0) && mask1;
        mask1 = (Q_(2, 0) < dist) && mask1;
        if(mask1) good1++;
        Triangulate(P0, P2, points1, points2, Q);
        bool mask2 = ((Q(2, 0) * Q(3, 0)) > 0);
        Q(0, 0) /= Q(3, 0);
        Q(1, 0) /= Q(3, 0);
        Q(2, 0) /= Q(3, 0);
        Q(3, 0) /= Q(3, 0);
        mask2 = ((Q(2, 0) < dist) && mask2);
        Q_ = P2 * Q;
        mask2 = (Q_(2, 0) > 0) && mask2;
        mask2 = (Q_(2, 0) < dist) && mask2;
        if(mask2) good2++;
        Triangulate(P0, P3, points1, points2, Q);
        bool mask3 = ((Q(2, 0) * Q(3, 0)) > 0);
        Q(0, 0) /= Q(3, 0);
        Q(1, 0) /= Q(3, 0);
        Q(2, 0) /= Q(3, 0);
        Q(3, 0) /= Q(3, 0);
        mask3 = ((Q(2, 0) < dist) && mask3);
        Q_ = P3 * Q;
        mask3 = (Q_(2, 0) > 0) && mask3;
        mask3 = (Q_(2, 0) < dist) && mask3;
        if(mask3) good3++;
        Triangulate(P0, P4, points1, points2, Q);
        bool mask4 = ((Q(2, 0) * Q(3, 0)) > 0);
        Q(0, 0) /= Q(3, 0);
        Q(1, 0) /= Q(3, 0);
        Q(2, 0) /= Q(3, 0);
        Q(3, 0) /= Q(3, 0); 
        mask4 = ((Q(2, 0) < dist) && mask4);
        Q_ = P4 * Q;
        mask4 = (Q_(2, 0) > 0) && mask4;
        mask4 = (Q_(2, 0) < dist) && mask4;
        if(mask4) good4++;
    }
    

    if (good1 >= good2 && good1 >= good3 && good1 >= good4) {
        _R = R1;
        _t = t;
        return good1;
    } else if (good2 >= good1 && good2 >= good3 && good2 >= good4) {
        _R = R2;
        _t = t;
        return good2;
    } else if (good3 >= good1 && good3 >= good2 && good3 >= good4) {
        _t = -t;
        _R = R1;
        return good3;
    } else {
        _t = -t;
        _R = R2;
        return good4;
    }
}