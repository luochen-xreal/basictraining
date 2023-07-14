
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "include/utils.h"
#include "include/extractandmatch.h"
#include "include/ceresBA.h"
#include "include/triangulation.h"
#include "include/solvepose.h"
#include <opencv2/features2d.hpp>
using namespace std;

using cv::Mat;
using std::vector;
int main(int argc, char const *argv[])
{
    Mat srcImg1 = cv::imread ("../two_image_pose_estimation/1403637188088318976.png");
    Mat srcImg2 = cv::imread ("../two_image_pose_estimation/1403637189138319104.png");

    vector<cv::KeyPoint> keypoint1, keypoint2;
    cv::Ptr<cv::SiftFeatureDetector> siftdtc = cv::SiftFeatureDetector::create(100);
    siftdtc->detect(srcImg1, keypoint1);
    siftdtc->detect(srcImg2, keypoint2);
    Mat keypointsImg;
    cv::drawKeypoints(srcImg1, keypoint1, keypointsImg);
    cv::imshow("keypoint Img", keypointsImg);


    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
    Mat descriptor1, descriptor2;
    extractor->compute(srcImg1, keypoint1, descriptor1);
    extractor->compute(srcImg2, keypoint2, descriptor2);
    cv::imshow("descriptor1", descriptor1);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

    vector<cv::DMatch> matches ;
    Mat imgMatches ;
    matcher->match ( descriptor1 , descriptor2, matches) ;
    
    cv::drawMatches ( srcImg1 , keypoint1 , srcImg2 , keypoint2, matches, imgMatches ) ;
    cv::imshow("matches", imgMatches);
    cv::waitKey(0);
    return 0;
}
