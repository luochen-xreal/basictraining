#include <opencv2/opencv.hpp>

int main() {
    cv::FileStorage fs("/home/xreal/xreal/two_image_pose_estimation/sensor.yaml", cv::FileStorage::READ);

    // Read sensor_type
    std::string sensor_type;
    fs["sensor_type"] >> sensor_type;

    // Read comment
    std::string comment;
    fs["comment"] >> comment;

    // Read T_BS matrix
    cv::Mat T_BS;
    fs["T_BS"] >> T_BS;

    // Read rate_hz
    int rate_hz;
    fs["rate_hz"] >> rate_hz;

    // Read resolution
    cv::Size resolution;
    fs["resolution"] >> resolution;

    // Read camera_model
    std::string camera_model;
    fs["camera_model"] >> camera_model;

    // Read intrinsics
    cv::Mat intrinsics;
    fs["intrinsics"] >> intrinsics;

    // Read distortion_model
    std::string distortion_model;
    fs["distortion_model"] >> distortion_model;

    // Read distortion_coefficients
    cv::Mat distortion_coefficients;
    fs["distortion_coefficients"] >> distortion_coefficients;

    // Print the read values
    std::cout << "sensor_type: " << sensor_type << std::endl;
    std::cout << "comment: " << comment << std::endl;
    std::cout << "T_BS matrix:\n" << T_BS << std::endl;
    std::cout << "rate_hz: " << rate_hz << std::endl;
    std::cout << "resolution: " << resolution << std::endl;
    std::cout << "camera_model: " << camera_model << std::endl;
    std::cout << "intrinsics matrix:\n" << intrinsics << std::endl;
    std::cout << "distortion_model: " << distortion_model << std::endl;
    std::cout << "distortion_coefficients matrix:\n" << distortion_coefficients << std::endl;

    // Close the file
    fs.release();

    return 0;
}
