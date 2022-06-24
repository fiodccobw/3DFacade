#include <iostream>
#include "json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "gurobi_c++.h"
#include "facadeModeling.h"
#include "reconstruction.h"
#include <easy3d/fileio/point_cloud_io.h>
#include<dirent.h>
#include "../3rd_party/ETH3DFormatLoader/images.h"
#include "../3rd_party/ETH3DFormatLoader/cameras.h"
#include "opencv2/calib3d/calib3d.hpp"
#include <easy3d/renderer/texture_manager.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <chrono>
using namespace std;

typedef std::vector<vector<vec3>> Hole;
class Box;
class Pt;


void triangulate(SurfaceMesh *mesh) {
    if (!mesh)
        return;

    mesh->update_face_normals();
    auto normals = mesh->face_property<vec3>("f:normal");
    auto holes = mesh->get_face_property<Hole>("f:holes");
    auto types = mesh->face_property<int>("f:types");
    Tessellator tessellator;
    int curnum = 0;
    vector<int> numlist;
    for (auto f : mesh->faces()) {
        int type = types[f];
        tessellator.begin_polygon(normals[f]);

        tessellator.set_winding_rule(Tessellator::WINDING_NONZERO);
        tessellator.begin_contour();
        for (auto h : mesh->halfedges(f)) {
            SurfaceMesh::Vertex v = mesh->target(h);
            tessellator.add_vertex(mesh->position(v), v.idx());
        }
        tessellator.end_contour();

        if (holes) { // has a valid hole

            for (auto p : holes[f]){
                tessellator.set_winding_rule(Tessellator::WINDING_ODD);
                tessellator.begin_contour();
                for(auto v:p){
                    tessellator.add_vertex(v);
                }
                tessellator.end_contour();
            }

        }

        tessellator.end_polygon();
        int totalnum = int(tessellator.elements().size()) - curnum;
        for(int i=0;i<totalnum;i++){
            numlist.push_back(type);
        }
        curnum = int(tessellator.elements().size());
    }

    // now the tessellation is done. We can clear the old mesh and
    // fill it will the new set of triangles

    mesh->clear();
    auto types2 = mesh->add_face_property<int>("f:types");
    const auto& triangles = tessellator.elements();

    if (!triangles.empty()) { // in degenerate cases num can be zero
        const std::vector<Tessellator::Vertex *> &vts = tessellator.vertices();
        for (auto v : vts) {
            mesh->add_vertex(vec3(v->data()));
        }
//        for (const auto& t : triangles) {
//
//        }
        for(int i=0;i<triangles.size();i++){
            SurfaceMesh::Face tri = mesh->add_triangle(SurfaceMesh::Vertex(triangles[i][0]), SurfaceMesh::Vertex(triangles[i][1]), SurfaceMesh::Vertex(triangles[i][2]));
            types2[tri] = numlist[i];
        }
    }
}

std::vector<Eigen::Vector4d> coord_Conversion(std::vector<Eigen::Vector3d> onplane, std::vector<Eigen::Vector3d> pts, Eigen::Vector3d normal,Eigen::Matrix4d& M) {
//    cout<<"A"<<endl;
    std::vector<Eigen::Vector4d> result;

    Eigen::Vector3d a = onplane[0], b = onplane[1], c = onplane[2];
    Eigen::Vector3d ab(b[0]-a[0], b[1] - a[1], b[2] - a[2]);
    Eigen::Vector3d ac(c[0]-a[0], c[1] - a[1], c[2] - a[2]);

        //conversion
        Eigen::Vector3d V = ab.normalized().cross(normal.normalized());
        Eigen::Vector3d u(ab.normalized()[0]+a[0],ab.normalized()[1]+a[1],ab.normalized()[2]+a[2]);
        Eigen::Vector3d v(a[0]+V[0],a[1]+V[1],a[2]+V[2]);
        Eigen::Vector3d n(normal.normalized()[0]+a[0],normal.normalized()[1]+a[1],normal.normalized()[2]+a[2]);
        Eigen::Matrix4d S;
        for(int i=0;i<4;i++){
            if(i<=2){
                S(i,0) = a[i];
                S(i,1) = u[i];
                S(i,2) = v[i];
                S(i,3) = n[i];
            }else{
                S(i,0) = 1;
                S(i,1) = 1;
                S(i,2) = 1;
                S(i,3) = 1;
            }
        }
        Eigen::Matrix4d DD;
        DD.setZero();
        DD(0,1) = 1;
        DD(1,2) = 1;
        DD(2,3) = 1;
        DD(3,0) = 1;
        DD(3,1) = 1;
        DD(3,2) = 1;
        DD(3,3) = 1;
        M = DD*S.inverse();
        for(auto i:pts){
            Eigen::Vector4d temp(i[0],i[1],i[2],1);
            auto newpt = M*temp;
            result.push_back(newpt);
//            std::cout<<"x:"<<newpt[0]<<" y:"<<newpt[1]<<" z:"<<newpt[2]<<" w:"<<newpt[3]<<std::endl;
        }
        return result;
}


void intrusion(SurfaceMesh* mesh,vector<Eigen::Vector3d> list1, vector<Eigen::Vector3d> list2, int ty){

    int t = ty;


    auto types = mesh->face_property<int>("f:types");
    std::vector<SurfaceMesh::Vertex> vertices = {
            mesh->add_vertex(vec3(float(list1[0][0]), float(list1[0][1]), float(list1[0][2]))),
            mesh->add_vertex(vec3(float(list2[0][0]), float(list2[0][1]), float(list2[0][2]))),
            mesh->add_vertex(vec3(float(list2[3][0]), float(list2[3][1]), float(list2[3][2]))),
            mesh->add_vertex(vec3(float(list1[3][0]), float(list1[3][1]), float(list1[3][2])))
    };
    SurfaceMesh::Face ff =mesh->add_face(vertices);
    types[ff] = t;
    std::vector<SurfaceMesh::Vertex> vertices1 = {
            mesh->add_vertex(vec3(float(list1[1][0]), float(list1[1][1]), float(list1[1][2]))),
            mesh->add_vertex(vec3(float(list2[1][0]), float(list2[1][1]), float(list2[1][2]))),
            mesh->add_vertex(vec3(float(list2[0][0]), float(list2[0][1]), float(list2[0][2]))),
            mesh->add_vertex(vec3(float(list1[0][0]), float(list1[0][1]), float(list1[0][2])))
    };
    SurfaceMesh::Face ff2 =mesh->add_face(vertices1);
    types[ff2] = t;
    std::vector<SurfaceMesh::Vertex> vertices2 = {
            mesh->add_vertex(vec3(float(list1[2][0]), float(list1[2][1]), float(list1[2][2]))),
            mesh->add_vertex(vec3(float(list2[2][0]), float(list2[2][1]), float(list2[2][2]))),
            mesh->add_vertex(vec3(float(list2[1][0]), float(list2[1][1]), float(list2[1][2]))),
            mesh->add_vertex(vec3(float(list1[1][0]), float(list1[1][1]), float(list1[1][2])))
    };
    SurfaceMesh::Face ff3 =mesh->add_face(vertices2);
    types[ff3] = t;
    std::vector<SurfaceMesh::Vertex> vertices3 = {
            mesh->add_vertex(vec3(float(list1[3][0]), float(list1[3][1]), float(list1[3][2]))),
            mesh->add_vertex(vec3(float(list2[3][0]), float(list2[3][1]), float(list2[3][2]))),
            mesh->add_vertex(vec3(float(list2[2][0]), float(list2[2][1]), float(list2[2][2]))),
            mesh->add_vertex(vec3(float(list1[2][0]), float(list1[2][1]), float(list1[2][2])))
    };
    SurfaceMesh::Face ff4 =mesh->add_face(vertices3);
    types[ff4] = t;
}

int main() {

    int step = 2;
    int ransac_param = 100000;
    int fix;
    const char * folder = "data/twofaces/images";
    string folderstr = "data/twofaces/";
    int numc;
    if(step == 1){
        auto start = std::chrono::high_resolution_clock::now();
        //#1 reconstruction
        PointCloud* cloud = PointCloudIO::load(folderstr+"/fused.ply");
        if (!cloud) {
            LOG(ERROR) << "Error: failed to load model. Please make sure the file exists and format is correct.";
            return EXIT_FAILURE;
        }
        std::cout << "point cloud has " << cloud->n_vertices() << " points" << std::endl;
        Reconstruction re;
        std::cout << "------Recontructing the coarse model------- "<< std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        SurfaceMesh* reconstructed = re.apply(cloud,ransac_param);
        auto finish1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = finish1 - start1;
        std::cout << "------------Elapsed time for reconstruction------: " << elapsed1.count() << " s\n";
        delete cloud;


        //#2 select images


        struct dirent *entry;
        DIR *dir = opendir(folder);
        std::vector<string> imagenamelist;
        while((entry = readdir(dir)) != NULL){
            imagenamelist.push_back(entry->d_name);
        }
        closedir(dir);


        //2.1 read the camera parameters
        ColmapCameraPtrMap cameramap;
        bool success = ReadColmapCameras(folderstr+"cameras.txt", &cameramap);
        if (success) {
            cout<<"read the intrinsic parameters"<<endl;
        } else {
            std::cout << "Error: could not load cameras." << std::endl;
        }
        ColmapImagePtrMap imagemap;
        bool success2 = ReadColmapImages(folderstr+"images.txt", /* read_observations */ false,
                                         &imagemap);
        if (success2) {
            cout<<"read the extrinsic parameters"<<endl;
        } else {
            std::cout << "Error: could not load image infos." << std::endl;
        }

        auto it = imagenamelist.begin();
        while(it != imagenamelist.end()){
            if(*it == "." || *it == ".."){
                it = imagenamelist.erase(it);
            }
            else{
                ++it;
            }
        }
        std::vector<std::vector<double>> intrinsiclist;
        std::vector<Eigen::Transform<float, 3, Eigen::Affine>> extrinsiclist_global2img;
        std::vector<Eigen::Transform<float, 3, Eigen::Affine>> extrinsiclist_img2global;
        std::vector<cv::Vec3f> rotationvectorlist;
        std::vector<cv::Mat> cameramatrixlist;
        std::vector<cv::Vec3f> translationvectorlist;
        std::vector<cv::Mat> rotationmatrixlist;
        std::vector<cv::Point3f> centerlist;
        std::vector<Eigen::Vector3f> xrotlist;
        std::vector<Eigen::Vector3f> zrotlist;
        for(auto name:imagenamelist){
            for(auto& v:imagemap){
                if(v.second->file_path == name){
                    extrinsiclist_global2img.push_back(v.second->image_T_global);
                    extrinsiclist_img2global.push_back(v.second->global_T_image);
//                for(auto& vv:c){
//                    if(vv.first == v.second->camera_id){
//                        intrinsiclist.push_back(vv.second->parameters);
//                    }
//                }
                    intrinsiclist.push_back(cameramap[v.second->camera_id]->parameters);
                }

            }
        }

        for(int ix=0;ix<intrinsiclist.size();ix++){
            cv::Mat cameramatrix;
            cv::Vec3f rotationvector;

            Eigen::Matrix3f rm = extrinsiclist_global2img[ix].rotation();
            cv::Mat rotationmatrix = (cv::Mat_<float>(3,3)<<rm(0,0),rm(0,1),rm(0,2),rm(1,0),rm(1,1),rm(1,2),rm(2,0),rm(2,1),rm(2,2));
            cv::Vec3f translationvector(extrinsiclist_global2img[ix].translation()[0],extrinsiclist_global2img[ix].translation()[1],extrinsiclist_global2img[ix].translation()[2]);
            cv::Mat translationmat = (cv::Mat_<float>(3,1)<<translationvector[0],translationvector[1],translationvector[2]);
            cv::Rodrigues(rotationmatrix,rotationvector);
//        cout<<"rotationvector: "<<rotationvector<<endl;
            cameramatrix = (cv::Mat_<float>(3,3)<<intrinsiclist[ix][0],0.0,intrinsiclist[ix][2],0.0,intrinsiclist[ix][1],intrinsiclist[ix][3],0.0,0.0,1.0);
            cv::Mat cameracenter = -rotationmatrix.inv()*translationmat;
            Eigen::Vector3f one(1,0,0);
            Eigen::Vector3f one2(0,0,1);
//        Eigen::Matrix3f rmt = extrinsiclist_img2global[ix].rotation();
//        auto vec = rmt*one;
            auto vec = rm.transpose()*one;
            auto vec2 = rm.transpose()*one2;

//        cout<<"trans2: "<<vec<<endl;

            xrotlist.push_back(vec);
            zrotlist.push_back(vec2);
            centerlist.push_back(cv::Point3f(cameracenter.at<float>(0,0),cameracenter.at<float>(0,1),cameracenter.at<float>(0,2)));
            rotationvectorlist.push_back(rotationvector);
            translationvectorlist.push_back(translationvector);
            cameramatrixlist.push_back(cameramatrix);
            rotationmatrixlist.push_back(rotationmatrix);
        }

        cout<<"\n";
        auto start2 = std::chrono::high_resolution_clock::now();
        std::cout << "-------------Selecting images-----------"<< std::endl;

        //2.2 calculate the metrics to select images
        auto vertex = reconstructed->vertex_property<vec3>("v:point");
        auto faces = reconstructed->faces_size();
        cout<<"number of faces: "<<faces<<endl;
            //2.2.1 reverse one face model to the camera's direction
        bool flag = false;
        if(faces == 1){
            for(auto face : reconstructed->faces()) {
                std::vector<cv::Point3f> fourpoints;
                for (auto p: reconstructed->vertices(face)) {
                    cv::Point3f vt(vertex[p][0], vertex[p][1], vertex[p][2]);
                    fourpoints.push_back(vt);
                }

                cv::Point3f centroidplane((fourpoints[1].x+fourpoints[2].x+fourpoints[3].x+fourpoints[0].x)/4,(fourpoints[1].y+fourpoints[2].y+fourpoints[3].y+fourpoints[0].y)/4,(fourpoints[1].z+fourpoints[2].z+fourpoints[3].z+fourpoints[0].z)/4);
//                cout<<"centroid: "<<centroidplane<<endl;

                cv::Vec3f vec1 = fourpoints[1] - fourpoints[0];
                cv::Vec3f vec2 = fourpoints[2] - fourpoints[1];
                cv::Vec3f normalvec = vec1.cross(vec2);
                Eigen::Vector3f  normal(normalvec[0],normalvec[1],normalvec[2]);
                normal = normal.normalized();
                cv::Vec3f normalcv(normal[0],normal[1],normal[2]);
                int num = 0;
                float sum = 0;
                for(auto c:centerlist){
                    cv::Vec3f connectedline = c-centroidplane;
                    Eigen::Vector3f  connectednorm(connectedline[0],connectedline[1],connectedline[2]);
                    connectednorm = connectednorm.normalized();
                    cv::Vec3f connectedlinecv(connectednorm[0],connectednorm[1],connectednorm[2]);
//                cout<<"camera center: "<<connectedlinecv<<endl;
                    float dotpro = connectedlinecv.dot(normalcv);
//                cout<<"dot production: "<<dotpro<<endl;
                    sum+=dotpro;
                    num++;
                }
//                cout<<"hereeeeeeeeeeeeeeeeee:"<<sum/float(num)<<endl;
                if(sum/float(num)<0){
                    flag = true;
                }

            }
            if(flag){
//                cout<<"reverssssssssss!"<<endl;
                reconstructed->reverse_orientation();

            }

        }
        SurfaceMeshIO::save("model/reconstructed.ply",reconstructed);
        //start the selection
        vector<string> selectedimagenamelist;
        vector<vector<float>> offsetlist;
        auto types = reconstructed->face_property<int>("f:types");
        for(auto face : reconstructed->faces()){
            cout<<"a!"<<endl;
            if(types[face] == 4){
                cout<<"processing face........................"<<endl;
                std::vector<cv::Point3f> fourpoints; //face corners in 3d
                for(auto p:reconstructed->vertices(face)){
                    cv::Point3f vt(vertex[p][0],vertex[p][1],vertex[p][2]);
//                    cout<<"vertex: "<<vt<<endl;
                    fourpoints.push_back(vt);
                }

                cv::Point3f centroidplane((fourpoints[1].x+fourpoints[2].x+fourpoints[3].x+fourpoints[0].x)/4,(fourpoints[1].y+fourpoints[2].y+fourpoints[3].y+fourpoints[0].y)/4,(fourpoints[1].z+fourpoints[2].z+fourpoints[3].z+fourpoints[0].z)/4);
//                cout<<"centroid: "<<centroidplane<<endl;

                cv::Vec3f vec1 = fourpoints[1] - fourpoints[0];
                cv::Vec3f vec2 = fourpoints[2] - fourpoints[1];

                //get the W/H ratio and the orthogonal face's normal
                float ratio;
                float sumdot1=0;
                float sumdot2=0;
                float count = 0;
                Eigen::Vector3f orthnormal;
                Eigen::Vector3f line1(vec1[0],vec1[1],vec1[2]);
                Eigen::Vector3f line2(vec2[0],vec2[1],vec2[2]);

                for(auto r:xrotlist){
                    float dot1 = r.normalized().dot(line1.normalized());
                    float dot2 = r.normalized().dot(line2.normalized());
                    sumdot1+=abs(dot1);
                    sumdot2+=abs(dot2);
                    count++;
                }
                if((sumdot1/float(count))>(sumdot2/float(count))){
                    ratio = line1.norm()/line2.norm();
                    orthnormal = line1;
                }else{
                    ratio = line2.norm()/line1.norm();
                    orthnormal = line2;
                }

                cv::Vec3f normalvec = vec1.cross(vec2);
                Eigen::Vector3f  normal(normalvec[0],normalvec[1],normalvec[2]);
                normal = normal.normalized();
                cv::Vec3f normalcv(normal[0],normal[1],normal[2]);

                float curdot = INFINITY;
                int curindex = 0;
                vector<cv::Point> curfp;
                float curxmin;
                float curymin;
                cv::Mat largeimg;

                for(int im=0;im<imagenamelist.size();im++) {
//                    cout<<imagenamelist[curindex]<<endl;
//
                    //3d points project to the image
                    vector<vector<cv::Point2f>> fourpoints_image;
                    vector<cv::Point2f> fpimg; // face corners in 2d
                    cv::Mat dis;
//                    cout<<imagenamelist[im]<<endl;
                    cv::projectPoints(fourpoints, rotationvectorlist[im], translationvectorlist[im], cameramatrixlist[im], dis,
                                      fpimg);

                    cv::Mat ci = cv::imread(folderstr+"/images/"+imagenamelist[im]);
                    float ww = float(ci.size().width);
                    float hh = float(ci.size().height);
                    bool condition1 = fpimg[0].x<0 && fpimg[1].x<0 && fpimg[2].x<0&&fpimg[3].x<0;
                    bool condition2 = fpimg[0].x>ww && fpimg[1].x>ww && fpimg[2].x>ww&&fpimg[3].x>ww;

                    if(!(condition1||condition2)){
                        //computer the intersection area and percentage
                        vector<cv::Point2f> fourcorner;
                        fourcorner.push_back(cv::Point2f(0, 0));
                        fourcorner.push_back(cv::Point2f(cameramap[1]->width, 0));
                        fourcorner.push_back(cv::Point2f(cameramap[1]->width, cameramap[1]->height));
                        fourcorner.push_back(cv::Point2f(0, cameramap[1]->height));

                        vector<float> xlist;
                        vector<float> ylist;
                        for (auto p: fpimg) {
                            xlist.push_back(p.x);
                            ylist.push_back(p.y);
                        }
                        xlist.push_back(0);
                        xlist.push_back(cameramap[1]->width);
                        ylist.push_back(0);
                        ylist.push_back(cameramap[1]->height);
                        float xmax = max(xlist);
                        float ymax = max(ylist);
                        float xmin = min(xlist);
                        float ymin = min(ylist);
                        cv::Mat newpic(ymax - ymin + 1,xmax - xmin + 1, CV_8UC1, cv::Scalar(0, 0, 0)); // new extended image * 2
                        cv::Mat newpic2(ymax - ymin + 1,xmax - xmin + 1, CV_8UC1, cv::Scalar(0, 0, 0));
//                    cv::Mat ci = cv::imread(folderstr+"/images/"+imagenamelist[im]);
                        cv::Mat cli;
//                    float ww = float(ci.size().width);
//                    float hh = float(ci.size().height);
//                    cv::copyMakeBorder(ci,cli,abs(ymin),abs(ymax- hh),abs(xmin),abs(xmax-ww),cv::BORDER_CONSTANT,0);
//                    cout<<"x min: "<<xmin<<" y min:"<<ymin<<endl;
                        vector<cv::Point> newfpimg; // face corners in 2d
                        for (auto p: fpimg) {
                            newfpimg.push_back(cv::Point(p.x - xmin, p.y - ymin));
                        }
                        cv::fillPoly(newpic, newfpimg, cv::Scalar(255, 0, 0), cv::LINE_8);

                        vector<cv::Point> newfourcorner; //image corners
                        for(auto p:fourcorner){
                            newfourcorner.push_back(cv::Point(p.x - xmin, p.y - ymin));
                        }

                        cv::fillPoly(newpic2, newfourcorner, cv::Scalar(255, 255, 255), cv::LINE_8);
//                        cout<<"hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii!"<<endl;
                        cv::Mat output;
                        cv::bitwise_and(newpic,newpic2,output);
                        cv::Mat nott;
                        cv::bitwise_not(output,nott);
//                    imwrite("rectified/"+imagenamelist[im],output);
//                    imwrite("rectified/a"+imagenamelist[im],newpic);

                        int originarea = cv::countNonZero(newpic);
                        int intersectarea = cv::countNonZero(output);
                        float percentage = float(intersectarea)/float(originarea);
//                    cout<<"name: "<<imagenamelist[im]<<" per: "<<percentage<<endl;
                        //compute the dot production between normalproj and connectedproj
                        float dotpro;

                        float dot = normal.dot(zrotlist[im].normalized());

                        if((percentage>0.9) && (dot<0)){
                            cv::Vec3f connectedline = centerlist[im]-centroidplane;
                            Eigen::Vector3f  connectednorm(connectedline[0],connectedline[1],connectedline[2]);
                            dotpro = orthnormal.normalized().dot(connectednorm.normalized());
                            if(abs(dotpro) < curdot){
                                curdot = abs(dotpro);
                                curindex = im;
                                curfp = newfpimg;
                                curxmin = xmin;
                                curymin = ymin;
                                largeimg = cli;
//                        cv::fillPoly(newpic, curfp, cv::Scalar(255, 255, 255), cv::LINE_8);
//                        cv::fillPoly(newpic, newfourcorner, cv::Scalar(255, 0, 255), cv::LINE_8);
//                        cv::imwrite("/home/fiodccob/GEO2020/regularization/results/result11.jpg",newpic);
                            }
                        }
                    }else{
                        continue;
                    }

                }

                cout<<"ratio: "<<ratio<<endl;
                cout<<"rotation: "<<curdot<<endl;
                cout<<"best images: "<<imagenamelist[curindex]<<endl;
                selectedimagenamelist.push_back(imagenamelist[curindex]);
                vector<float> templistos;
                templistos.push_back(curxmin);
                templistos.push_back(curymin);
                offsetlist.push_back(templistos);
                size_t lastindex = imagenamelist[curindex].find_last_of(".");
                string rawname = imagenamelist[curindex].substr(0, lastindex);
                cv::FileStorage fs("rectified/"+rawname+"_cameraparam.yml", cv::FileStorage::WRITE);
                fs<<"cm"<<cameramatrixlist[curindex];
//            fs.release();
                fs<<"tv"<<translationvectorlist[curindex];
                fs<<"rm"<<rotationmatrixlist[curindex];
                fs.release();

                cv::Mat curimg = cv::imread(folderstr+"/images/"+imagenamelist[curindex]);
                facadeModeling f;
//                cv::imwrite("yolov5/data/samples2/larger"+imagenamelist[curindex],largeimg);
//                for(auto p:curfp){
////                    cout<<"current four points: "<<p.x<<" "<<p.y<<endl;
//                }


                f.recification(curimg,imagenamelist[curindex],ratio,curfp);

            }

        }
        auto finish2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = finish2 - start2;
        std::cout << "------------Elapsed time for image selection and rectification------: " << elapsed2.count() << " s\n";
        ofstream fsys("rectified/imagelist.txt");
        for(auto n:selectedimagenamelist){
            fsys<<n<<endl;
        }
//        ofstream fsys2("rectified/offsetlist.txt");
//        for(auto n:offsetlist){
//            fsys2<<n[0]<<" "<<n[1]<<endl;
//        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "------------Elapsed time------: " << elapsed.count() << " s\n";
    }
    else if(step == 2){

        cout<<"--------------IMPLEMENT STEP 2----------------"<<endl;
        //#backprojection
        SurfaceMesh* reconstructed = SurfaceMeshIO::load("model/reconstructed.ply");
//        SurfaceMesh* reconstructed_temp = SurfaceMeshIO::load("model/reconstructed.ply");
//        auto texcoord = mesh->add_vertex_property<vec2>("v:texcoord");

        auto vertex = reconstructed->vertex_property<vec3>("v:point");
        auto faces = reconstructed->faces_size();
        reconstructed->update_face_normals();
        auto normals = reconstructed->face_property<vec3>("f:normal");
        auto holes = reconstructed->add_face_property<Hole>("f:holes");
        auto types = reconstructed->face_property<int>("f:types");
        auto numfaces = reconstructed->faces_size();
        std::ifstream file("rectified/imagelist.txt");
        vector<std::string> imagenamelist;
        string str;
        while (std::getline(file, str))
        {
            imagenamelist.push_back(str);
        }

//        vector<string>
//        for(auto im:imagenamelist){
//
//
//        }

        int index = 0;

        for(auto face : reconstructed->faces()){
            if(types[face] == 4){
                auto start = std::chrono::high_resolution_clock::now();
                //            cout<<"type: "<<types[face]<<endl;
                std::vector<cv::Point3f> fourpoints;
                std::vector<Eigen::Vector3d> fourpoints3d;
                for (auto p: reconstructed->vertices(face)) {
                    cv::Point3f vt(vertex[p][0], vertex[p][1], vertex[p][2]);
                    Eigen::Vector3d v(vertex[p][0], vertex[p][1], vertex[p][2]);
                    fourpoints.push_back(vt);
                    fourpoints3d.push_back(v);
                }


                size_t lastindex = imagenamelist[index].find_last_of(".");
                string rawname = imagenamelist[index].substr(0, lastindex);

                //get parameters
                cv::Mat tempimg = cv::imread("objectdetection/predictions/"+imagenamelist[index]);
                cv::Mat timg = cv::imread("rectified_img/"+imagenamelist[index]);

                double scale_x = 0.3;
                double scale_y = 0.3;
                cv::Mat scaled;
                cv::resize(tempimg,scaled,cv::Size(),scale_x,scale_y,cv::INTER_LINEAR);
                cv::imshow("image"+ to_string(index),scaled);
                auto finish = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = finish - start;
                std::cout << "------------Elapsed time1------: " << elapsed.count() << " s\n";

                cv::waitKey(0);
                cout<<"please input a number(1 for fix the layout,0 for not fix):"<<endl;
                cin>>fix;
                cout<<"please input the number of clusters in the facade:"<<endl;
                cin>>numc;
                auto start0 = std::chrono::high_resolution_clock::now();
                //read cameramatrix
                cv::FileStorage cp("rectified/"+rawname+"_cameraparam.yml", cv::FileStorage::READ);
                cv::Mat cameramatrix;
                cv::Vec3f translationvector;
                cv::Mat rotationmatrix;
                cp["cm"] >> cameramatrix;
                cp["tv"] >> translationvector;
                cp["rm"] >> rotationmatrix;
                cv::Mat translationmat = (cv::Mat_<float>(3,1)<<translationvector[0],translationvector[1],translationvector[2]);
                cv::Mat cameracenter = -rotationmatrix.inv()*translationmat;
                cv::Point3f cc(cameracenter.at<float>(0,0),cameracenter.at<float>(0,1),cameracenter.at<float>(0,2));
//            Eigen::Vector3f center(cc.x,cc.y,cc.z);
                cp.release();
                cv::Mat transformmatrix;
                cv::FileStorage transf("rectified/"+rawname+".yml",cv::FileStorage::READ);
                transf["transform mat"]>>transformmatrix;
                transf.release();
                cv::Mat image = cv::imread(folderstr+"/images/"+imagenamelist[index]);
                float width = float(image.size().width);
                float height = float(image.size().height);
                //calculate the plane equation
                cv::Vec3f vec1 = fourpoints[1] - fourpoints[0];
                cv::Vec3f vec2 = fourpoints[2] - fourpoints[1];
                cv::Vec3f normalvec = vec1.cross(vec2);
                Eigen::Vector3f  normal(normalvec[0],normalvec[1],normalvec[2]); //plane normal(point to the outside of the mesh)
                normal = normal.normalized();
                Eigen::Vector3d  normal3d(normal[0],normal[1],normal[2]);
                float a = normal[0];
                float b = normal[1];
                float c = normal[2];
                float d = -(a*fourpoints[0].x+b*fourpoints[0].y+c*fourpoints[0].z);
                open3d::geometry::PointCloud npc;
                Eigen::Vector3d center(cc.x,cc.y,cc.z);
                npc.points_.push_back(center);

                //find points in the fused point clouds within a threshold distance between the plane
                double threshold = 0.5;
                open3d::geometry::PointCloud cloud;
                open3d::io::ReadPointCloud(folderstr+"/fused.ply",cloud);
                double divider = a*a+b*b+c*c;
                vector<Eigen::Vector3d> pointsaround;
                for(auto p:cloud.points_){
                    double divided = (a*p[0]+b*p[1]+c*p[2]+d)*(a*p[0]+b*p[1]+c*p[2]+d);
                    double distsqr = divided/divider;
                    if(distsqr<(threshold*threshold)){
                        pointsaround.push_back(p);
                    }
                }

                cv::Vec3f rotationvector;
                cv::Rodrigues(rotationmatrix,rotationvector);
                cv::Mat dis;
                vector<cv::Point2f> fpimg; // face corners in 2d
                //get the corner points in 2d and then transform to rectification image
                cv::projectPoints(fourpoints, rotationvector, translationvector, cameramatrix, dis,
                                  fpimg);

                vector<cv::Point2f> newfpimg; //transformed face corners
                cv::perspectiveTransform(fpimg,newfpimg,transformmatrix);


//                TextureManager::request()





//                cv::Mat image2 = cv::imread("rectified_img/"+imagenamelist[index]);
                vector<cv::Point> fff;
                for(auto p:newfpimg){
                    cv::Point np(p.x,p.y);
                    fff.push_back(np);
                }
//                cv::fillPoly(image2, fff, cv::Scalar(255, 255, 255), cv::LINE_8);
//                cv::imwrite("rectified_img/projected2.jpg",image2);


                auto start1 = std::chrono::high_resolution_clock::now();

                facadeModeling F;
                cout<<"------------REGULARIZATION---------"<<endl;

                vector<Box> windowlist = F.regularization(imagenamelist[index],0, fix,newfpimg,numc);
                cout<<"size: "<<windowlist.size()<<endl;
                auto finish1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed1 = finish1 - start1;
                std::cout << "------------Elapsed time for regularization------: " << elapsed1.count() << " s\n";
                if(windowlist.empty()){
                    index++;
                    continue;
                }


//            open3d::geometry::PointCloud newpc;
//            for(auto i:pointsaround){
//                newpc.points_.push_back(i);
//            }
//            open3d::io::WritePointCloudToPLY("model/pointsaround"+std::to_string(index)+".ply",newpc,true);

//                cout<<"----------------------------------points inside threshold: "<<pointsaround.size()<<endl;
                Eigen::Matrix4d M;
//            cout<<"---------------depth:   "<<normal3d<<endl;
                vector<Eigen::Vector4d> converted;
//            cout<<"21222: "<<fourpoints3d.size()<<endl;
                converted = coord_Conversion(fourpoints3d,pointsaround,normal3d,M);

                vector<float> depthlist;
                vector<vector<Eigen::Vector3d>> ptliststored;
                vector<Box> newboxlist;
                vector<Eigen::Vector4d> fourptnew; // converted four face corners
                for(auto p:fourpoints3d){
                    Eigen::Vector4d pp(p[0],p[1],p[2],1);
                    auto newp = M*pp;
                    fourptnew.push_back(newp);
                }
                for(auto box:windowlist){
                    int count =0;
                    double depthsum = 0;
                    vector<cv::Point2f> plist;
                    vector<cv::Point2f> newplist;
                    cv::Point2f p1(box.x,box.y);
                    plist.push_back(p1);
                    cv::Point2f p2(box.x+box.width,box.y);
                    plist.push_back(p2);
                    cv::Point2f p3(box.x+box.width,box.y+box.height);
                    plist.push_back(p3);
                    cv::Point2f p4(box.x,box.y+box.height);
                    plist.push_back(p4);

                    cv::perspectiveTransform(plist,newplist,transformmatrix.inv()); // get the original coordinate of the boxes using the transform matrix

                    vector<cv::Point3f> plist3d;
                    //back projection
                    for(auto p:newplist){
                        cv::Mat uvPoint = cv::Mat::ones(3,1,cv::DataType<float>::type);
                        uvPoint.at<float>(0,0)  = p.x;
                        uvPoint.at<float>(1,0) = p.y;
                        cv::Mat tvec;
                        tvec.create(3,1,cv::DataType<float>::type);
                        tvec.at<float>(0,0) = translationvector[0];
                        tvec.at<float>(1,0) = translationvector[1];
                        tvec.at<float>(2,0) = translationvector[2];
                        cv::Mat tempMat, tempMat2;
                        float s=0, zConst = 1;

                        tempMat = rotationmatrix.inv() * cameramatrix.inv() * uvPoint;

                        tempMat2 = rotationmatrix.inv() * tvec;

                        s = zConst + tempMat2.at<float>(2,0);
                        s /= tempMat.at<float>(2,0);

                        cv::Mat wcPoint = rotationmatrix.inv() * (s * cameramatrix.inv() * uvPoint - tvec);
                        cv::Point3f realPoint(wcPoint.at<float>(0, 0), wcPoint.at<float>(1, 0), wcPoint.at<float>(2, 0));
                        plist3d.push_back(realPoint);
//                    cout<<"point: "<<p<<endl;
                    }

                    //find the intersection between the line and the facade plane
                    vector<Eigen::Vector3d> newptlist;

                    double xmin1 = fourptnew[0][0];
                    double xmax1 = fourptnew[1][0];
                    double ymin1 = fourptnew[1][1];
                    double ymax1 = fourptnew[2][1];
                    if(xmin1 > xmax1){
                        double tmp = xmax1;
                        xmax1 = xmin1;
                        xmin1 = tmp;
                    }
                    if(ymin1 > ymax1){
                        double tmp = ymax1;
                        ymax1 = ymin1;
                        ymin1 = tmp;
                    }

                    //calculate the intersection point
                    for(auto p:plist3d){
                        Eigen::Vector3d pt(p.x,p.y,p.z);
                        Eigen::Vector3d linenormal = pt - center;
                        linenormal = linenormal.normalized();
                        //line parameters
                        auto a1 = float(linenormal[0]);
                        auto b1 = float(linenormal[1]);
                        auto c1 = float(linenormal[2]);
                        auto up = -d-a*cc.x-b*cc.y-c*cc.z;
                        auto down = a*a1+b*b1+c*c1;
                        auto t = up/down;
                        Eigen::Vector3d newpt(cc.x+a1*t,cc.y+b1*t,cc.z+c1*t);
                        newptlist.push_back(newpt);
//                    npc.points_.push_back(newpt);
                    }

                    //3d points

                    vector<Eigen::Vector4d> convertednewptlist; //2d points(4)
                    for(auto p:newptlist){
                        Eigen::Vector4d pp(p[0],p[1],p[2],1);
                        auto newp = M*pp;
                        convertednewptlist.push_back(newp);
                    }
                    bool flaag = true;
                    for(auto p:convertednewptlist){
                        if(p[0]>xmin1 && p[0]<xmax1 && p[1]>ymin1 && p[1]<ymax1){
                            continue;
                        }else{
                            flaag = false;
                            break;
                        }
                    }

                    if(flaag){
                        vector<double> xlist,ylist;
                        for(auto p:convertednewptlist){
                            xlist.push_back(p[0]);
                            ylist.push_back(p[1]);
                        }
                        double xmin = min(xlist);
                        double xmax = max(xlist);
                        double ymin = min(ylist);
                        double ymax = max(ylist);
                        if(xmin > xmax){
                            double tmp = xmax;
                            xmax = xmin;
                            xmin = tmp;
                        }
                        if(ymin > ymax){
                            double tmp = ymax;
                            ymax = ymin;
                            ymin = tmp;
                        }
//                    cout<<"xmin: "<<xmin<<" xmax: "<<xmax<<" ymin: "<<ymin<<" ymax: "<<ymax<<endl;
                        for(auto p:converted){
//                            cout<<"x: "<<p[0]<<" y: "<<p[1]<<endl;
                            if(p[0]>=xmin && p[0]<=xmax && p[1]>=ymin && p[1]<=ymax ){
//                                cout<<"!!!!"<<endl;
                                count++;
                                depthsum+=p[2];
                            }
                        }

                        double depth = depthsum/float(count);
                        depthlist.push_back(depth);
                        ptliststored.push_back(newptlist);
                        newboxlist.push_back(box);
//                    cout<<"---------------depth:   "<<count<<endl;
                    }

                }
                bool flag = true;
                int type = 0;
                vector<float> depthbytype;
                while(flag){
                    float dsum=0;
                    float count = 0;
                    bool flag2 = false;
                    for(int i=0;i<newboxlist.size();i++){
//                    cout<<"new type: "<< newboxlist[i].type<<endl;
                        if(newboxlist[i].type == type){
                            dsum += depthlist[i];
                            count++;
                            flag2 = true;
                        }
                    }
                    if(flag2){
                        float dep = dsum/count;
//                    cout<<"dsum:"<<dsum<<" count: "<<count<<endl;
                        depthbytype.push_back(dep);
                    }else{
                        depthbytype.push_back(0);
                    }
                    if(type == 2){
                        flag = false;
                    }
                    type++;
                }
//                cout<<"---------------depth by type:   "<<depthbytype.size()<<endl;

                for(int i=0;i<ptliststored.size();i++){
                    vector<Eigen::Vector3d> newptlist = ptliststored[i];
//                cv::fillPoly(image,plistint,cv::Scalar(255, 0, 255), cv::LINE_8);
//                for(auto p:newptlist){
//                    std::vector<SurfaceMesh::Vertex> vertices = {
//                            mesh->add_vertex(vec3(float(newptlist[0][0]), float(newptlist[0][1]), float(newptlist[0][2]))),
//                            mesh->add_vertex(vec3(float(newptlist[1][0]), float(newptlist[1][1]), float(newptlist[1][2]))),
//                            mesh->add_vertex(vec3(float(newptlist[2][0]), float(newptlist[2][1]), float(newptlist[2][2]))),
//                            mesh->add_vertex(vec3(float(newptlist[3][0]), float(newptlist[3][1]), float(newptlist[3][2])))
//                    };
//                mesh->add_face(vertices);
                    //add holes
                    holes[face].push_back({
                                                  vec3(float(newptlist[0][0]), float(newptlist[0][1]), float(newptlist[0][2])),
                                                  vec3(float(newptlist[1][0]), float(newptlist[1][1]), float(newptlist[1][2])),
                                                  vec3(float(newptlist[2][0]), float(newptlist[2][1]), float(newptlist[2][2])),
                                                  vec3(float(newptlist[3][0]), float(newptlist[3][1]), float(newptlist[3][2]))

                                          });

                    vector<Eigen::Vector4d> convertedbox;
                    for(auto p:newptlist){
                        Eigen::Vector4d pp(p[0],p[1],p[2],1);
                        auto newp = M*pp;
                        convertedbox.push_back(newp);
                    }
                    //calculate the depth

//                    double depth = depthbytype[newboxlist[i].type];
                        double depth = 0.1;
//                    cout<<"cur depth: "<<depth<<" type:"<<newboxlist[i].type<<endl;
                    if((abs(depth)<0.005 || depth>=0 || isnan(depth))&&newboxlist[i].type == 0){
                        depth = -0.1;
                    }
                    if(newboxlist[i].type == 1 && (depth<0 || isnan(depth))){
                        depth = 0.1;
                    }
                    if(newboxlist[i].type == 2){
                        depth=-0.08;
                    }
//                cout<<"depth:"<<depth<<endl;
                    vector<Eigen::Vector3d> tlist;
                    for(auto p:convertedbox){
                        Eigen::Vector4d newp(p[0],p[1],depth,1);
                        Eigen::Vector4d newpp = M.inverse()*newp;
                        Eigen::Vector3d newppp(newpp[0],newpp[1],newpp[2]);
                        tlist.push_back(newppp);
                    }
                    std::vector<SurfaceMesh::Vertex> vertices = {
                            reconstructed->add_vertex(vec3(float(tlist[0][0]), float(tlist[0][1]), float(tlist[0][2]))),
                            reconstructed->add_vertex(vec3(float(tlist[3][0]), float(tlist[3][1]), float(tlist[3][2]))),
                            reconstructed->add_vertex(vec3(float(tlist[2][0]), float(tlist[2][1]), float(tlist[2][2]))),
                            reconstructed->add_vertex(vec3(float(tlist[1][0]), float(tlist[1][1]), float(tlist[1][2])))
                    };
                    SurfaceMesh::Face ff = reconstructed->add_face(vertices);
                    types[ff] = newboxlist[i].type;
                    int tyyy = newboxlist[i].type;
                    intrusion(reconstructed,tlist,newptlist,tyyy);

                }

                cout<<"hole length: "<<holes[face].size()<<endl;

                index++;
                //make it as a building block
                if(numfaces==1){

                    vector<Eigen::Vector3d> clist;
                    for(auto p:fourptnew){
                        Eigen::Vector4d newp(p[0],p[1],-5,1);
                        Eigen::Vector4d newpp = M.inverse()*newp;
                        Eigen::Vector3d newppp(newpp[0],newpp[1],newpp[2]);
                        clist.push_back(newppp);
                    }
                    std::vector<SurfaceMesh::Vertex> vertices1 = {
                            reconstructed->add_vertex(vec3(float(clist[0][0]), float(clist[0][1]), float(clist[0][2]))),
                            reconstructed->add_vertex(vec3(float(clist[3][0]), float(clist[3][1]), float(clist[3][2]))),
                            reconstructed->add_vertex(vec3(float(clist[2][0]), float(clist[2][1]), float(clist[2][2]))),
                            reconstructed->add_vertex(vec3(float(clist[1][0]), float(clist[1][1]), float(clist[1][2])))
                    };
                    SurfaceMesh::Face ff = reconstructed->add_face(vertices1);
                    types[ff] = 4;
                    intrusion(reconstructed,clist,fourpoints3d,4);
                }
                auto finish0 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed0 = finish0 - start0;
                std::cout << "------------Elapsed time2------: " << elapsed0.count() << " s\n";
            }


        }

        triangulate(reconstructed);
        SurfaceMeshIO::save("model/finalmesh.ply",reconstructed);



    }

}
