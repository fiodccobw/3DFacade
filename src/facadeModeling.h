//
// Created by fiodccob on 23-03-22.
//

#ifndef LAYOUT_FACADEMODELING_H
#define LAYOUT_FACADEMODELING_H

#endif //LAYOUT_FACADEMODELING_H

#pragma once
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "json.hpp"
#include "gurobi_c++.h"
#include "../3rd_party/ETH3DFormatLoader/images.h"
#include "../3rd_party/ETH3DFormatLoader/cameras.h"
#include<dirent.h>
#include "utils.h"
class facadeModeling{
public:
    Mat3<float> intrinsicParam;
    Mat3<float> rotation;
    Mat<3,1,float> translation;
    ColmapCameraPtrMap camerastxt;
    ColmapImagePtrMap imagestxt;
    cv::Mat img;
    facadeModeling();
    cv::Mat recification(cv::Mat img, string imgname,float ratio, vector<cv::Point> newfourcorner);
    static void click_event(int event, int x, int y, int flags, void *param);
    vector<Box> regularization(string imagename, int mode, int fix, vector<cv::Point2f> fpimg, int numf);
    vector<cv::Vec4i> edgedetection(cv::Mat img, vector<Box> windowlist,int i);
};