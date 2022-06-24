//
// Created by fiodccob on 21-03-22.
//

#ifndef LAYOUT_UTILS_H
#define LAYOUT_UTILS_H


#endif //LAYOUT_UTILS_H

#pragma once
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/surface_mesh_builder.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/algo/tessellator.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/stop_watch.h>
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/camera.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace easy3d;

class Box{
public:
    int id;
    float x;
    float y;
    float width;
    float height;
    float midx;
    float midy;
    float area;
    float ratio;
    int type;
    int visited = 0;
    Box(int id, float x, float y, float width, float height,int type):id(id),x(x),y(y),width(width),height(height),midx(x + width/2.0),midy(y + height/2.0),area(width*height),ratio(height/width),type(type){}
    void update(float x, float y, float width, float height);

};

class Pt{
public:
    float x;
    float y;
    Pt(float x, float y):x(x),y(y){}
};

SurfaceMesh * readMesh(const string& name);

Mat3<float> readIntrinsic(const string& path, const string& name);

Mat<3,4,float> readExtrinsic(const string& path, const string& name);

double max(std::vector<double> list);

float max(std::vector<float> list);

double min(std::vector<double> list);


float min(std::vector<float> list);

int max_index(std::vector<double> list);

int min_index(std::vector<double> list);


std::vector<float> solveFunction_2var(float num1x, float num1y, float sum1, float num2x, float num2y, float sum2);

