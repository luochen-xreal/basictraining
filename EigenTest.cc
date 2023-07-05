#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace std;
using namespace Eigen;
int main(int argc, char const *argv[])
{
    Quaterniond imu_q_caml(0.99090224973327068,
                            0.13431639597354814,
                            0.00095051670014565813,
                            -0.0084222184858180373);
                            
    Quaterniond imu_q_camr(0.99073762672679389,
                            0.13492462817073628,
                            -0.00013648999867379373,
                            -0.015306242884176362);

    Vector3d imu_p_caml( -0.050720060477640147,
                        -0.0017414170413474165,
                        0.0022943667597148118);

    Vector3d imu_p_camr(0.051932496584961352,
                        -0.0011555929083120534,
                        0.0030949732069645722);
    Quaterniond caml_q_camr = imu_q_caml.inverse() * imu_q_camr;
    Vector3d caml_p_camr = imu_q_caml.inverse() * imu_p_camr - imu_q_caml.inverse() * imu_p_caml;
    std::cout << "q: " << caml_q_camr.x() << "  " << caml_q_camr.y() << "  " <<  caml_q_camr.z() << "  " << caml_q_camr.w() << std::endl; 
    std::cout << "p:" << caml_p_camr.transpose() << std::endl; 


    // block

    MatrixXf m(4, 4);
    m<< 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;     

    cout << m.block<2, 2>(1, 1) << endl << endl;
    for(int i = 1; i <= 3; i++){
        cout << "block of size" << i << "x" << i << endl;
        cout << m.block(0, 0, i, i) << endl << endl;
    }

    Array22f a;
    a << 1, 2, 3, 4;
    m.block<2, 2>(1, 1) = a;
    cout << m << endl << endl;
    
    // 边角矩阵
    cout << "m.leftCols(2) =" << endl << m.leftCols(2) << endl << endl;
    cout << "m.topRows(2) =" << endl << m.topRows(3) << endl << endl;
    cout << "m.bottomRows(2) =" << endl << m.bottomRows(1) << endl << endl;
    cout << "m.topLeftCorner(1, 2) =" << endl << m.topLeftCorner(1, 2) << endl << endl;
    cout << "m.bottomLeftCorner(2, 1) =" << endl << m.bottomLeftCorner(2, 1) << endl << endl;


    // 向量

    ArrayXf v(6);
    v << 1, 2, 3, 4, 5, 6;
    cout << "v.head(3)" << endl << v.head(3) << endl << endl;
    cout << "v.tail(2)" << endl << v.tail(2) << endl << endl;
    cout << "v.segment(1, 3)" << endl << v.segment(1, 3) << endl << endl;
    cout << "v.segment<3>(1)" << endl << v.segment<3>(1) << endl << endl;
    return 0;
}
