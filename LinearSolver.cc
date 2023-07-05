#include <iostream>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;
using namespace Eigen;


bool ReadAndSolveDirect(string txtfile){
    std::ifstream file(txtfile); // 打开文本文件
    
    std::vector<double> x;
    std::vector<double> y_noise;
    x.reserve(100);
    y_noise.reserve(100);
    cout << "--------------------" << endl;
    if (file.is_open()) {
        std::string line;
        cout << "open file " << txtfile << endl;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double num1, num2;

            // 尝试从每一行中提取两个数字
            if (iss >> num1 >> num2) {
                x.push_back(num1);
                y_noise.push_back(num2);
            }
        }
        file.close(); // 关闭文件
        // y = mx + n;
        Vector2d mn;
        MatrixXd A(x.size(), 2);
        VectorXd b(x.size());
        MatrixXd A_plus(2, x.size());
        
        for(size_t i = 0; i < x.size(); i++){
            A.row(i) << x[i], 1;
            b.row(i) << y_noise[i];
        }
        A_plus = (A.transpose() * A).inverse() * A.transpose();
        // cout << "A:" << endl << A << endl;
        // cout << "b:" << endl << b << endl;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd1(A);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A_plus);
        double cond1 = svd1.singularValues()(0) * svd2.singularValues()(0); 
        double cond2 = svd1.singularValues()(0) / svd1.singularValues()(svd1.singularValues().size()-1); 
        cout << "condition1 of A:" << cond1 << endl;
        cout << "condition2 of A:" << cond2 << endl;
        cout << "The solution using the QR decomposition is:" << endl << A.colPivHouseholderQr().solve(b) << endl;
        cout << "The solution using normal equations is:" << endl << (A.transpose() * A).ldlt().solve(A.transpose() * b) << endl;
        cout << "The least-squares solution is:" << endl << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
    } else {
        std::cout << "无法打开文件" << std::endl;
        return false;
    }
    cout << "--------------------" << endl;
    return true;
}   


bool ReadAndSolve(string txtfile){
    std::ifstream file(txtfile); // 打开文本文件
    
    std::vector<double> x;
    std::vector<double> y_noise;
    x.reserve(100);
    y_noise.reserve(100);
    cout << "--------------------" << endl;
    if (file.is_open()) {
        std::string line;
        cout << "open file " << txtfile << endl;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double num1, num2;

            // 尝试从每一行中提取两个数字
            if (iss >> num1 >> num2) {
                x.push_back(num1);
                y_noise.push_back(num2);
            }
        }
        file.close(); // 关闭文件
        // y = mx + n;
        Vector2d mn;
        mn << 3, 1;
        double sum_xx = 0, sum_x = 0, sum_xy = 0, sum_y = 0;
        for(size_t i = 0;  i < x.size() ; i++){
            sum_xx += x[i] * x[i];
            sum_x += x[i];
            sum_xy += x[i] * y_noise[i];
            sum_y += y_noise[i];
        }
        double n = x.size();
        MatrixXd A(2, 2);
        A << sum_xx, sum_x, sum_x, n;
        VectorXd b(2);
        b << sum_xy, sum_y;
        cout << "A:" << endl << A << endl;
        cout << "b:" << endl << b << endl;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
        double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1); 
        cout << "condition of A:" << cond << endl;
        cout << "The solution using the QR decomposition is:" << endl << A.colPivHouseholderQr().solve(b) << endl;
        cout << "The solution using normal equations is:" << endl << (A.transpose() * A).ldlt().solve(A.transpose() * b) << endl;
        cout << "The least-squares solution is:" << endl << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;


        
    } else {
        std::cout << "无法打开文件" << std::endl;
        return false;
    }
    cout << "--------------------" << endl;
    return true;
}   

int main() {
    ReadAndSolveDirect("../data.txt");
    ReadAndSolveDirect("../data2.txt");
    return 0;
}


