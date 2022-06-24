//
// Created by fiodccob on 24-03-22.
//

#ifndef LAYOUT_RECONSTRUCTION_H
#define LAYOUT_RECONSTRUCTION_H

#endif //LAYOUT_RECONSTRUCTION_H
#pragma once
#include <easy3d/algo/point_cloud_ransac.h>
#include <easy3d/algo/point_cloud_normals.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/viewer/viewer.h>
#include <easy3d/algo/point_cloud_normals.h>
#include <easy3d/core/random.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "open3d/Open3D.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.h"
using namespace easy3d;


class Reconstruction{
public:
    Reconstruction();
    ~Reconstruction();
    SurfaceMesh* apply(PointCloud* pc, int ransac_thre);
    Eigen::Vector3d fitplane(std::vector<Eigen::Vector3d> pts,Eigen::Vector3d& cent);
    std::vector<Eigen::Vector3d> project_Plane(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d normal, Eigen::Vector3d &cent);
    std::vector<Eigen::Vector4d> coord_Conversion(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d normal,Eigen::Matrix4d& M);
    std::vector<Eigen::Vector3d> project_Line(std::vector<Eigen::Vector3d> pts1, std::vector<Eigen::Vector3d> pts2, Eigen::Vector3d direction, Eigen::Vector3d point_on_line );
    std::vector<Eigen::Vector3d> project_Line(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d direction, Eigen::Vector3d point_on_line );
    std::vector<int> find_Endpoints(std::vector<Eigen::Vector3d> projected, Eigen::Vector3d point_on_line, Eigen::Vector3d direction);
};

