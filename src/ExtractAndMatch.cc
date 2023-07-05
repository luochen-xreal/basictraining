
#include "include/ExtractAndMatch.h"
using namespace std;



void featureMatchingRANSAC(cv::Mat& img1, cv::Mat& img2,
                           std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                           std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matchesRANSAC,
                           int featureCount, double ransacThreshold) {
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(featureCount);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    cv::Mat descriptors1, descriptors2;
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    matcher->match(descriptors1, descriptors2, matches);

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    std::vector<uchar> inliers(points1.size(), 0);
    cv::Mat fundamental = cv::findFundamentalMat(points1, points2, inliers, cv::FM_RANSAC, ransacThreshold);

    for (size_t i = 0; i < inliers.size(); i++) {
        if (inliers[i]) {
            matchesRANSAC.push_back(matches[i]);
        }
    }
}

int main(int argc, char const *argv[])
{
    cv::Mat image1, image2;
    return 0;
}
