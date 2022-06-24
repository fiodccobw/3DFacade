//
// Created by fiodccob on 24-03-22.
//



#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/fileio/surface_mesh_io.h>

#include "reconstruction.h"



struct Node{
public:
    float x;
    float y;
    Node* first;
    Node* second;
    bool visited = false;
    Node(float x, float y):x(x),y(y),first(nullptr),second(nullptr){};
};

struct Line{
public:
    Eigen::Vector3d pt1;
    Eigen::Vector3d pt2;
    Eigen::Vector2d pt2d1;
    Eigen::Vector2d pt2d2;
    Node* n1 = nullptr;
    Node* n2 = nullptr;
    bool pt1updated = false;
    bool pt2updated = false;
    Line(Eigen::Vector3d pt1,Eigen::Vector3d pt2):pt1(pt1),pt2(pt2){};
};

Reconstruction::Reconstruction() {

}

Reconstruction::~Reconstruction() {

}

SurfaceMesh* Reconstruction::apply(PointCloud *cloud, int ransac_thre){

    //RANSAC first
    PointCloudNormals normal;
    normal.estimate(cloud);
    auto* ransac = new PrimitivesRansac();
    ransac->add_primitive_type(PrimitivesRansac::PLANE);
    int num = ransac->detect(cloud,ransac_thre);
    delete ransac;
    auto types = cloud->get_vertex_property<int>("v:primitive_index");
    auto points = cloud->get_vertex_property<vec3>("v:point");

    //remove outliers based on the RANSAC result
    open3d::geometry::PointCloud npc;
    std::vector<std::vector<Eigen::Vector3d>> ptss;
    // vector stores points list in different planes
    for(int i = 0; i<num;i++){
        npc.points_.clear();
        std::vector<Eigen::Vector3d> list;
        ptss.push_back(list);
        for(auto v: cloud->vertices()){
            if(types[v] == i){
                Eigen::Vector3d pt(points[v][0],points[v][1],points[v][2]);
                npc.points_.push_back(pt);
            }
        }
          for(auto v:npc.points_){
              ptss[i].push_back(v);
          }
    }


    //DO THE EXTRUSION(2 OR MORE FACADES) OR CLIP THE FITTED PLANE(ONE PLANE)
    //fit plane
    if(num == 1){
        Eigen::Vector3d centroid(0,0,0);
//        Eigen::Vector3d normal_vec =  fitplane(ptss[0],centroid);
        Eigen::Vector3d normal_vec =  fitplane(ptss[0],centroid);
        std::vector<Eigen::Vector3d> newpt0 = project_Plane(ptss[0],normal_vec,centroid);
        open3d::geometry::PointCloud newpc0;
        for(auto i:newpt0){
            newpc0.points_.push_back(i);
        }

        std::tuple< std::shared_ptr< open3d::geometry::PointCloud >, std::vector< size_t > > result = newpc0.RemoveStatisticalOutliers(10,2,true);
//        std::cout<<std::get<0>(result)->points_.size()<<std::endl;
        open3d::geometry::PointCloud newpc;
        std::vector<Eigen::Vector3d> newpt;
        for(auto v:std::get<0>(result)->points_){
            newpc.points_.push_back(v);
            newpt.push_back(v);
        }



        open3d::io::WritePointCloudToPLY("model/initialization.ply",newpc,true);

        //1. fit ONE plane
        Eigen::Matrix4d M;
        std::vector<Eigen::Vector4d> convertedpts = coord_Conversion(newpt,normal_vec,M);
        std::vector<cv::Point2f> plist;
        for(auto p:convertedpts){
            cv::Point2f pt;
            pt.x = float(p[0]);
            pt.y = float(p[1]);
            plist.push_back(pt);
        }

        //minimum bounding box
        cv::Point2f vtx[4];
        cv::RotatedRect box = cv::minAreaRect(plist);
        box.points(vtx);


        Eigen::Vector4d tl(vtx[0].x,vtx[0].y,0,1);
        Eigen::Vector4d tr(vtx[1].x,vtx[1].y,0,1);
        Eigen::Vector4d bl(vtx[2].x,vtx[2].y,0,1);
        Eigen::Vector4d br(vtx[3].x,vtx[3].y,0,1);
        auto backtl = M.inverse()*tl;
        auto backtr = M.inverse()*tr;
        auto backbl = M.inverse()*bl;
        auto backbr = M.inverse()*br;


        //output to mesh
        SurfaceMesh *reconstructed = new SurfaceMesh;
        auto tlist = reconstructed->add_face_property<int>("f:types");
        std::vector<SurfaceMesh::Vertex> vertices = {
                reconstructed->add_vertex(vec3(float(backbl[0]), float(backbl[1]), float(backbl[2]))),
                reconstructed->add_vertex(vec3(float(backbr[0]), float(backbr[1]), float(backbr[2]))),
                reconstructed->add_vertex(vec3(float(backtl[0]), float(backtl[1]), float(backtl[2]))),
                reconstructed->add_vertex(vec3(float(backtr[0]), float(backtr[1]), float(backtr[2])))
        };
        SurfaceMesh::Face ff = reconstructed->add_face(vertices);
        tlist[ff] = 4;
        return reconstructed;
    }else{

        //2. fit MORE planes
        //2.1. get the building height
        std::vector<Eigen::Vector3d> normallist; //fitted plane normals
        std::vector<std::vector<Eigen::Vector3d>> ptlist; //projected fitted plane points list

            //fit and projection
        for(int i = 0;i<num;i++){
            Eigen::Vector3d centroid(0,0,0);
            Eigen::Vector3d normal_vec =  fitplane(ptss[i],centroid);
            std::vector<Eigen::Vector3d> newpt0 = project_Plane(ptss[i],normal_vec,centroid);

            open3d::geometry::PointCloud newpc0;
            for(auto ii:newpt0){
                newpc0.points_.push_back(ii);
            }
            std::tuple< std::shared_ptr< open3d::geometry::PointCloud >, std::vector< size_t > > result = newpc0.RemoveStatisticalOutliers(10,2,true);
//            std::cout<<std::get<0>(result)->points_.size()<<std::endl;
            open3d::geometry::PointCloud newpc;
            std::vector<Eigen::Vector3d> newpt;
            for(auto v:std::get<0>(result)->points_){
                newpc.points_.push_back(v);
                newpt.push_back(v);
            }


            open3d::io::WritePointCloudToPLY("model/initialization" + std::to_string(i) + ".ply", newpc, true);


            normallist.push_back(normal_vec);
            ptlist.push_back(newpt);
        }

            //get height and the bottom plane
        float height = -INFINITY;
        Eigen::Vector3d bottom_normal;
        Eigen::Vector3d endpt;
        Eigen::Vector3d sumcross(0,0,0);
        vector<Eigen::Vector3d> crosslist;
        Eigen::Vector3d sumendpt(0,0,0);
        int count0 = 0;
        float sumheight = 0;

        cout<<"normal size: "<<normallist.size()<<endl;
        for(int i=0;i<normallist.size();i++){
            for(int ii = i+1; ii<normallist.size();ii++){
                Eigen::Vector3d crossproduct = normallist[i].normalized().cross(normallist[ii].normalized());
                    if(crossproduct.norm()>0.5){
                    int index1 = rand() % ptlist[i].size();
                    int index2 = rand() % ptlist[ii].size();

                    Eigen::Vector3d point1 = ptlist[i][10];
                    Eigen::Vector3d point2 = ptlist[ii][20];
                    double d1 = -(normallist[i][0]*point1[0]+normallist[i][1]*point1[1]+normallist[i][2]*point1[2]);
                    double d2 = -(normallist[ii][0]*point2[0]+normallist[ii][1]*point2[1]+normallist[ii][2]*point2[2]);
                    std::vector<float> xy = solveFunction_2var(float(normallist[i][0]),float(normallist[i][1]),-float(d1),float(normallist[ii][0]),float(normallist[ii][1]),-float(d2));

                    Eigen::Vector3d point_on_line(xy[0],xy[1],0);
                    std::vector<Eigen::Vector3d> projected = project_Line(ptlist[i],ptlist[ii],crossproduct,point_on_line);
                    open3d::geometry::PointCloud newpc;
                    std::vector<int> endpoints = find_Endpoints(projected, point_on_line, crossproduct);
                    Eigen::Vector3d tempep;
                    if(projected[endpoints[0]][2]<projected[endpoints[1]][2]){
                        tempep = projected[endpoints[0]];
                    }else{
                        tempep = projected[endpoints[1]];
                    }
                    float cur_height = float((projected[endpoints[0]] - projected[endpoints[1]]).norm());

                    sumheight+=cur_height;
                    crosslist.push_back(crossproduct);
                    double c4 = sumendpt[0];
                    double c5 = sumendpt[1];
                    double c6 = sumendpt[2];
                    sumendpt = Eigen::Vector3d(c4+tempep[0],c5+tempep[1],c6+tempep[2]);
                    count0++;
                }


            }
        }

        height = sumheight/float(count0);
        endpt = Eigen::Vector3d(sumendpt[0]/count0,sumendpt[1]/count0,sumendpt[2]/count0);
        double sum1=0,sum2=0,sum3=0;
        int countt=0;
        for(int i=0;i<crosslist.size()-1;i++){
            for(int jj=i+1;jj<crosslist.size();jj++){
                if(crosslist[i].dot(crosslist[jj])<0.1){
                    crosslist[jj] = -crosslist[jj];
                }
            }
        }
        for(auto v:crosslist){
            sum1 += v[0];
            sum2 += v[1];
            sum3 += v[2];
            countt++;
        }
        bottom_normal = Eigen::Vector3d(sum1/countt,sum2/countt,sum3/countt);


        //2.2. project to the intersected line between the bottom plane and planes
        std::vector<std::vector<Eigen::Vector3d>> projectedlines_bottom;
        std::vector<Eigen::Vector3d> directionlist;
        for(int i = 0;i<num;i++){
            Eigen::Vector3d crossproduct = normallist[i].cross(bottom_normal);
            Eigen::Vector3d point1 = ptlist[i][0];

            double d1 = -(normallist[i][0]*point1[0]+normallist[i][1]*point1[1]+normallist[i][2]*point1[2]);
            double d2 = -(bottom_normal[0]*endpt[0]+bottom_normal[1]*endpt[1]+bottom_normal[2]*endpt[2]);
            std::vector<float> xy = solveFunction_2var(float(normallist[i][0]),float(normallist[i][1]),-float(d1),float(bottom_normal[0]),float(bottom_normal[1]),-float(d2));
            Eigen::Vector3d point_on_line(xy[0],xy[1],0);
            std::vector<Eigen::Vector3d> projected = project_Line(ptlist[i],crossproduct,point_on_line);
            open3d::geometry::PointCloud newpc;
            for(auto i:projected){
                newpc.points_.push_back(i);
            }

            projectedlines_bottom.push_back(projected);
            directionlist.push_back(crossproduct);
        }


        //2.3 get the graph
        //the conversion matrix M
        std::vector<Line> linelist;
        Eigen::Matrix4d M_bottom;
        std::vector<Eigen::Vector3d> projbottom;
        for(auto p:projectedlines_bottom){
            for(auto pp:p){
                projbottom.push_back(pp);
            }
        }
        coord_Conversion(projbottom,bottom_normal,M_bottom);

        //endpoints of the lines
        std::vector<double> lengthlist;
        for(int l = 0; l<projectedlines_bottom.size();l++){
            std::vector<int> endpoints = find_Endpoints(projectedlines_bottom[l],projectedlines_bottom[l][0],directionlist[l]);
            Eigen::Vector3d len = projectedlines_bottom[l][endpoints[0]] - projectedlines_bottom[l][endpoints[1]];
            lengthlist.push_back(len.norm());
            Line newl(projectedlines_bottom[l][endpoints[0]],projectedlines_bottom[l][endpoints[1]]);
            linelist.push_back(newl);
        }


        //convert coordinates
        for(auto& l:linelist){

            Eigen::Vector4d p1(l.pt1[0],l.pt1[1],l.pt1[2],1);
            Eigen::Vector4d p2(l.pt2[0],l.pt2[1],l.pt2[2],1);
            Eigen::Vector4d pt14d = M_bottom * p1;
            Eigen::Vector4d pt24d = M_bottom * p2;
            Eigen::Vector2d pt12d(pt14d[0],pt14d[1]);
            Eigen::Vector2d pt22d(pt24d[0],pt24d[1]);
            l.pt2d1 = pt12d;
            l.pt2d2 = pt22d;
        }


        //calculate distances
        std::vector<std::vector<double>> distlist1;
        std::vector<std::vector<double>> distlist2;
        std::vector<std::vector<int>> indexlist1;
        std::vector<std::vector<int>> indexlist2;
        for(int i=0; i<linelist.size();i++){
            std::vector<double> d1;
            std::vector<double> d2;
            std::vector<int> idx1;
            std::vector<int> idx2;

            for(int ii = 0;ii<linelist.size();ii++){
                if(i != ii) {
                    Eigen::Vector2d pair1 = linelist[i].pt2d1-linelist[ii].pt2d1;
                    Eigen::Vector2d pair2 = linelist[i].pt2d1-linelist[ii].pt2d2;
                    Eigen::Vector2d pair3 = linelist[i].pt2d2-linelist[ii].pt2d1;
                    Eigen::Vector2d pair4 = linelist[i].pt2d2-linelist[ii].pt2d2;
                    double dist1 = pair1.norm();
                    double dist2 = pair2.norm();
                    double dist3 = pair3.norm();
                    double dist4 = pair4.norm();
                    d1.push_back(dist1);
                    d1.push_back(dist2);
                    d2.push_back(dist3);
                    d2.push_back(dist4);
                    idx1.push_back(ii);
                    idx1.push_back(ii);
                    idx2.push_back(ii);
                    idx2.push_back(ii);
                }
            }
            distlist1.push_back(d1);
            distlist2.push_back(d2);
            indexlist1.push_back(idx1);
            indexlist2.push_back(idx2);
        }


        std::vector<Node*> nodelist;
        std::vector<pair<int,int>> idxpairs;
        std::vector<int> indicator;
        std::vector<int> inds;
        double minlength = min(lengthlist);
        for(int i=0;i<linelist.size();i++){

            if(!linelist[i].pt1updated){
                int index1 = min_index(distlist1[i]);
                double minvalue1 = min(distlist1[i]);
                if(minvalue1<(minlength/2)){
                        double k1 = (linelist[i].pt2d1[1]-linelist[i].pt2d2[1]) / (linelist[i].pt2d1[0]-linelist[i].pt2d2[0]);
                        double k2 = (linelist[indexlist1[i][index1]].pt2d1[1]-linelist[indexlist1[i][index1]].pt2d2[1]) / (linelist[indexlist1[i][index1]].pt2d1[0]-linelist[indexlist1[i][index1]].pt2d2[0]);
                        double sum1 = k1*linelist[i].pt2d1[0] - linelist[i].pt2d1[1];
                        double sum2 = k2*linelist[indexlist1[i][index1]].pt2d1[0] - linelist[indexlist1[i][index1]].pt2d1[1];
                        std::vector<float> intersectpt = solveFunction_2var(float(k1),-1,float(sum1),float(k2),-1,float(sum2));
                        Eigen::Vector2d initnode(intersectpt[0],intersectpt[1]);
                        Node* x = new Node(intersectpt[0],intersectpt[1]);
                        if(index1%2 == 0){
                            linelist[i].pt2d1 = initnode;
                            linelist[indexlist1[i][index1]].pt2d1 = initnode;
                            linelist[i].pt1updated = true;
                            linelist[indexlist1[i][index1]].pt1updated = true;
                            linelist[i].n1 = x;
                            linelist[indexlist1[i][index1]].n1 = x;
                            indicator.push_back(0);
                            nodelist.push_back(x);
                            idxpairs.push_back(make_pair(i,indexlist1[i][index1]));
                        }
                        if(index1%2 == 1){
                            linelist[i].pt2d1 = initnode;
                            linelist[indexlist1[i][index1]].pt2d2 = initnode;
                            linelist[i].pt1updated = true;
                            linelist[indexlist1[i][index1]].pt2updated = true;
                            linelist[i].n1 = x;
                            linelist[indexlist1[i][index1]].n2 = x;
                            indicator.push_back(1);
                            nodelist.push_back(x);
                            idxpairs.push_back(make_pair(i,indexlist1[i][index1]));
                        }

                }
            }
            if(!linelist[i].pt2updated){
                int index2 = min_index(distlist2[i]);
                double minvalue2 = min(distlist2[i]);
                if(minvalue2<(minlength/2)){
                    if(!linelist[i].pt2updated){
                        double k1 = (linelist[i].pt2d1[1]-linelist[i].pt2d2[1]) / (linelist[i].pt2d1[0]-linelist[i].pt2d2[0]);
                        double k2 = (linelist[indexlist2[i][index2]].pt2d1[1]-linelist[indexlist2[i][index2]].pt2d2[1]) / (linelist[indexlist2[i][index2]].pt2d1[0]-linelist[indexlist2[i][index2]].pt2d2[0]);
                        double sum1 = k1*linelist[i].pt2d1[0] - linelist[i].pt2d1[1];
                        double sum2 = k2*linelist[indexlist2[i][index2]].pt2d1[0] - linelist[indexlist2[i][index2]].pt2d1[1];
                        std::vector<float> intersectpt = solveFunction_2var(float(k1),-1,float(sum1),float(k2),-1,float(sum2));
                        Eigen::Vector2d initnode(intersectpt[0],intersectpt[1]);
                        Node* x = new Node(intersectpt[0],intersectpt[1]);
                        if(index2%2 == 0){
                            linelist[i].pt2d2 = initnode;
                            linelist[indexlist2[i][index2]].pt2d1 = initnode;
                            linelist[i].pt2updated = true;
                            linelist[indexlist2[i][index2]].pt1updated = true;
                            linelist[i].n2 = x;
                            linelist[indexlist2[i][index2]].n1 = x;
                            indicator.push_back(2);
                            nodelist.push_back(x);
                            idxpairs.push_back(make_pair(i,indexlist2[i][index2]));
                        }
                        if(index2%2 == 1){
                            linelist[i].pt2d2 = initnode;
                            linelist[indexlist2[i][index2]].pt2d2 = initnode;
                            linelist[i].pt2updated = true;
                            linelist[indexlist2[i][index2]].pt2updated = true;
                            linelist[i].n2 = x;
                            linelist[indexlist2[i][index2]].n2 = x;
                            indicator.push_back(3);
                            nodelist.push_back(x);
                            idxpairs.push_back(make_pair(i,indexlist2[i][index2]));
                        }

                    }

                }
            }
        }



        int count = 0;

        for(int l=0; l<linelist.size();l++){
            if(!linelist[l].pt1updated){
                count++;
                Node* n = new Node(float(linelist[l].pt2d1[0]),float(linelist[l].pt2d1[1]));
                n->first = linelist[l].n2;
                linelist[l].n1 = n;
                nodelist.push_back(n);
            }
            if(!linelist[l].pt2updated){
                count++;
                Node* n = new Node(float(linelist[l].pt2d2[0]),float(linelist[l].pt2d2[1]));
                n->first = linelist[l].n1;
                linelist[l].n2 = n;
                nodelist.push_back(n);
            }
        }



        //connect the nodes
        for(int i=0;i<nodelist.size()-count;i++){

            switch (indicator[i]) {
                case(0):
                    nodelist[i]->first = linelist[idxpairs[i].first].n2;
                    nodelist[i]->second = linelist[idxpairs[i].second].n2;
                    break;
                case(1):
                    nodelist[i]->first = linelist[idxpairs[i].first].n2;
                    nodelist[i]->second = linelist[idxpairs[i].second].n1;
                    break;
                case(2):
                    nodelist[i]->first = linelist[idxpairs[i].first].n1;
                    nodelist[i]->second = linelist[idxpairs[i].second].n2;
                    break;
                case(3):
                    nodelist[i]->first = linelist[idxpairs[i].first].n1;
                    nodelist[i]->second = linelist[idxpairs[i].second].n1;
                    break;
            }
        }



        std::vector<Eigen::Vector2d> finalshape;
        Node* cur_node;
        for(int i=0;i<nodelist.size();i++){
            if(nodelist[i]->second == nullptr){
                Eigen::Vector2d begin(nodelist[i]->x,nodelist[i]->y);
                finalshape.push_back(begin);
                cur_node = nodelist[i];
                break;
            }
        }

        if(finalshape.empty()){
            Eigen::Vector2d begin(nodelist[0]->x,nodelist[0]->y);
            finalshape.push_back(begin);
            cur_node = nodelist[0];
        }
        Node* prev = nullptr;
        Node* next = nullptr;
        while(finalshape.size() != nodelist.size()){
            if(finalshape.size() == 1){

                next = cur_node->first;
                Eigen::Vector2d p(next->x,next->y);
                finalshape.push_back(p);
                prev = cur_node;
                cur_node = next;
            }else{

                next = cur_node->first;
                if(next == prev){
                    next = cur_node->second;
                }
                Eigen::Vector2d p(next->x,next->y);
                finalshape.push_back(p);
                prev = cur_node;
                cur_node = next;
            }
        }

        //make it counterclockwise
        double sum = 0;
        for(int j = 0; j<finalshape.size();j++){
            if(j == finalshape.size()-1){
                double value = (finalshape[0][0]-finalshape[j][0])*(finalshape[0][1]+finalshape[j][1]);
            }
            double value = (finalshape[j+1][0]-finalshape[j][0])*(finalshape[j+1][1]+finalshape[j][1]);
            sum+=value;
        }
        if(sum>0)
            std::reverse(finalshape.begin(),finalshape.end());

        //convert to 3d and extrusion
        SurfaceMesh *reconstructed2 = new SurfaceMesh;
        auto tlist = reconstructed2->add_face_property<int>("f:types");
        for(int k=0;k<finalshape.size();k++){
            if(count == 0 && k == finalshape.size()-1){
//                cout<<111111<<endl;
                Eigen::Vector4d p1(finalshape[k][0],finalshape[k][1],0,1);
                Eigen::Vector4d p2(finalshape[0][0],finalshape[0][1],0,1);
                Eigen::Vector4d p3(finalshape[0][0],finalshape[0][1],-height,1);
                Eigen::Vector4d p4(finalshape[k][0],finalshape[k][1],-height,1);
                auto v1 = M_bottom.inverse()*p1;
                auto v2 = M_bottom.inverse()*p2;
                auto v3 = M_bottom.inverse()*p3;
                auto v4 = M_bottom.inverse()*p4;
                std::vector<SurfaceMesh::Vertex> vertices = {
                        reconstructed2->add_vertex(vec3(float(v1[0]), float(v1[1]), float(v1[2]))),
                        reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2]))),
                        reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2]))),
                        reconstructed2->add_vertex(vec3(float(v4[0]), float(v4[1]), float(v4[2])))
                };
                SurfaceMesh::Face ff = reconstructed2->add_face(vertices);
                tlist[ff] = 4;
            }else if(count>0 && k == finalshape.size()-1){
                break;
            }
            else{
                Eigen::Vector4d p1(finalshape[k][0],finalshape[k][1],0,1);
                Eigen::Vector4d p2(finalshape[k+1][0],finalshape[k+1][1],0,1);
                Eigen::Vector4d p3(finalshape[k+1][0],finalshape[k+1][1],-height,1);
                Eigen::Vector4d p4(finalshape[k][0],finalshape[k][1],-height,1);
                auto v1 = M_bottom.inverse()*p1;
                auto v2 = M_bottom.inverse()*p2;
                auto v3 = M_bottom.inverse()*p3;
                auto v4 = M_bottom.inverse()*p4;
                std::vector<SurfaceMesh::Vertex> vertices = {
                        reconstructed2->add_vertex(vec3(float(v1[0]), float(v1[1]), float(v1[2]))),
                        reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2]))),
                        reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2]))),
                        reconstructed2->add_vertex(vec3(float(v4[0]), float(v4[1]), float(v4[2])))
                };
                SurfaceMesh::Face ff = reconstructed2->add_face(vertices);
                tlist[ff] = 4;
            }
        }

        if(finalshape.size() == 3){
            double x = finalshape[2][0] - finalshape[1][0] + finalshape[0][0];
            double y = finalshape[2][1] - finalshape[1][1] + finalshape[0][1];
            Eigen::Vector4d newp(x,y,0,1);

            Eigen::Vector4d p1(finalshape[0][0],finalshape[0][1],0,1);
            Eigen::Vector4d p2(x,y,0,1);
            Eigen::Vector4d p3(x,y,-height,1);
            Eigen::Vector4d p4(finalshape[0][0],finalshape[0][1],-height,1);
            Eigen::Vector4d p5(finalshape[2][0],finalshape[2][1],0,1);
            Eigen::Vector4d p6(finalshape[2][0],finalshape[2][1],-height,1);
            Eigen::Vector4d p7(finalshape[1][0],finalshape[1][1],0,1);
            Eigen::Vector4d p8(finalshape[1][0],finalshape[1][1],-height,1);
            auto v1 = M_bottom.inverse()*p1;
            auto v2 = M_bottom.inverse()*p2;
            auto v3 = M_bottom.inverse()*p3;
            auto v4 = M_bottom.inverse()*p4;
            auto v5 = M_bottom.inverse()*p5;
            auto v6 = M_bottom.inverse()*p6;
            auto v7 = M_bottom.inverse()*p7;
            auto v8 = M_bottom.inverse()*p8;
            //face1
            std::vector<SurfaceMesh::Vertex> vertices = {
                    reconstructed2->add_vertex(vec3(float(v1[0]), float(v1[1]), float(v1[2]))),
                    reconstructed2->add_vertex(vec3(float(v4[0]), float(v4[1]), float(v4[2]))),
                    reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2]))),
                    reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2])))
            };
            SurfaceMesh::Face ff = reconstructed2->add_face(vertices);
            tlist[ff] = 5;
            std::vector<SurfaceMesh::Vertex> vertices1 = {
                    reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2]))),
                    reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2]))),
                    reconstructed2->add_vertex(vec3(float(v6[0]), float(v6[1]), float(v6[2]))),
                    reconstructed2->add_vertex(vec3(float(v5[0]), float(v5[1]), float(v5[2])))
            };
            SurfaceMesh::Face ff1 = reconstructed2->add_face(vertices1);
            tlist[ff1] = 5;
            std::vector<SurfaceMesh::Vertex> vertices2 = {
                    reconstructed2->add_vertex(vec3(float(v1[0]), float(v1[1]), float(v1[2]))),
                    reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2]))),
                    reconstructed2->add_vertex(vec3(float(v5[0]), float(v5[1]), float(v5[2]))),
                    reconstructed2->add_vertex(vec3(float(v7[0]), float(v7[1]), float(v7[2])))
            };
            SurfaceMesh::Face ff2 = reconstructed2->add_face(vertices2);
            tlist[ff2] = 5;
            std::vector<SurfaceMesh::Vertex> vertices3 = {
                    reconstructed2->add_vertex(vec3(float(v4[0]), float(v4[1]), float(v4[2]))),
                    reconstructed2->add_vertex(vec3(float(v8[0]), float(v8[1]), float(v8[2]))),
                    reconstructed2->add_vertex(vec3(float(v6[0]), float(v6[1]), float(v6[2]))),
                    reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2])))
            };
            SurfaceMesh::Face ff3 = reconstructed2->add_face(vertices3);
            tlist[ff3] = 5;
        }else if(finalshape.size()>3 && count>0){
            Eigen::Vector4d p1(finalshape[0][0],finalshape[0][1],0,1);
            Eigen::Vector4d p2(finalshape[0][0],finalshape[0][1],-height,1);
            Eigen::Vector4d p3(finalshape[finalshape.size()-1][0],finalshape[finalshape.size()-1][1],0,1);
            Eigen::Vector4d p4(finalshape[finalshape.size()-1][0],finalshape[finalshape.size()-1][1],-height,1);
            auto v1 = M_bottom.inverse()*p1;
            auto v2 = M_bottom.inverse()*p2;
            auto v3 = M_bottom.inverse()*p3;
            auto v4 = M_bottom.inverse()*p4;
            std::vector<SurfaceMesh::Vertex> vertices = {
                    reconstructed2->add_vertex(vec3(float(v1[0]), float(v1[1]), float(v1[2]))),
                    reconstructed2->add_vertex(vec3(float(v2[0]), float(v2[1]), float(v2[2]))),
                    reconstructed2->add_vertex(vec3(float(v4[0]), float(v4[1]), float(v4[2]))),
                    reconstructed2->add_vertex(vec3(float(v3[0]), float(v3[1]), float(v3[2])))
            };
            SurfaceMesh::Face ff = reconstructed2->add_face(vertices);
            tlist[ff] = 5;
            std::vector<SurfaceMesh::Vertex> vertices1;
            std::vector<SurfaceMesh::Vertex> vertices2;
            for(auto p:finalshape){
                Eigen::Vector4d newp1(p[0],p[1],-height,1);
                auto v0 = M_bottom.inverse()*newp1;
                SurfaceMesh::Vertex ver = reconstructed2->add_vertex(vec3(float(v0[0]), float(v0[1]), float(v0[2])));
                vertices1.push_back(ver);
            }
            SurfaceMesh::Face ff1 = reconstructed2->add_face(vertices1);
            tlist[ff1] = 5;
            for(int i=finalshape.size()-1;i>=0;i--){
                Eigen::Vector4d newp1(finalshape[i][0],finalshape[i][1],0,1);
                auto v0 = M_bottom.inverse()*newp1;
                SurfaceMesh::Vertex ver = reconstructed2->add_vertex(vec3(float(v0[0]), float(v0[1]), float(v0[2])));
                vertices2.push_back(ver);
            }
            SurfaceMesh::Face ff2 = reconstructed2->add_face(vertices2);
            tlist[ff2] = 5;
        }

        std::vector<vec3> finalpoints3d;
        return reconstructed2;
    }

}

Eigen::Vector3d Reconstruction::fitplane(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d& cent) {
    size_t num_atoms = pts.size();
    Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);
    for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = pts[i];

    // calculate centroid
    Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

    // subtract centroid
    coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

    // we only need the left-singular matrix here
    //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d plane_normal = svd.matrixU().rightCols(1);
    cent = centroid;
    return plane_normal;

}

std::vector<Eigen::Vector3d> Reconstruction::project_Plane(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d normal,Eigen::Vector3d& cent) {
    std::vector<Eigen::Vector3d> result;
    for(auto p:pts){
        Eigen::Vector3d dst;
        double t;
        t = (normal[0]*cent[0] + normal[1]*cent[1] + normal[2]*cent[2] - normal[0]*p[0] - normal[1]*p[1] - normal[2]*p[2]) / (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        dst[0] = p[0] + t * normal[0];
        dst[1] = p[1] + t * normal[1];
        dst[2] = p[2] + t * normal[2];
        result.push_back(dst);
    }
    return result;
}


std::vector<Eigen::Vector4d> Reconstruction::coord_Conversion(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d normal,Eigen::Matrix4d& M) {
    bool flag = true;
//    Eigen::Matrix4d M;
    std::vector<Eigen::Vector4d> result;
    while(flag){
        int index1 = rand() % pts.size();
        int index2 = rand() % pts.size();
        int index3 = rand() % pts.size();
        Eigen::Vector3d a = pts[index1], b = pts[index2], c = pts[index3];
        Eigen::Vector3d ab(b[0]-a[0], b[1] - a[1], b[2] - a[2]);
        Eigen::Vector3d ac(c[0]-a[0], c[1] - a[1], c[2] - a[2]);
        float test = abs((ab.dot(ac))/(ab.norm()*ac.norm()));
//        std::cout<<"test result:"<<test<<std::endl;
        if(test<0.9){
            flag = false;
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
            }
        }
    }



    return result;
}

std::vector<Eigen::Vector3d> Reconstruction::project_Line(std::vector<Eigen::Vector3d> pts1,std::vector<Eigen::Vector3d> pts2, Eigen::Vector3d direction, Eigen::Vector3d point_on_line) {

    std::vector<Eigen::Vector3d> result;
    for(auto p:pts1){
        Eigen::Vector3d ap = p - point_on_line;
        Eigen::Vector3d pproject = point_on_line + (ap.dot(direction)*direction)/(direction.norm()*direction.norm());
        result.push_back(pproject);
    }    for(auto p:pts2){
        Eigen::Vector3d ap = p - point_on_line;
        Eigen::Vector3d pproject = point_on_line + (ap.dot(direction)*direction)/(direction.norm()*direction.norm());
        result.push_back(pproject);
    }
    return result;
}

std::vector<int> Reconstruction::find_Endpoints(std::vector<Eigen::Vector3d> projected, Eigen::Vector3d point_on_line, Eigen::Vector3d direction) {
    float maxp = -INFINITY;
    float minp = INFINITY;
    int maxindex = 0;
    int minindex = 0;
    for(int i =0; i< projected.size();i++){
        double t = (projected[i][0]-point_on_line[0]) / direction[0];
        if(t>maxp){
            maxp = float(t);
            maxindex = i;
        }
        if(t<minp){
            minp = float(t);
            minindex = i;
        }
    }
    std::vector<int> result;
    result.push_back(maxindex);
    result.push_back(minindex);

    return result;
}

std::vector<Eigen::Vector3d> Reconstruction::project_Line(std::vector<Eigen::Vector3d> pts, Eigen::Vector3d direction, Eigen::Vector3d point_on_line) {
    std::vector<Eigen::Vector3d> result;
    for(auto p:pts){
        Eigen::Vector3d ap = p - point_on_line;
        Eigen::Vector3d pproject = point_on_line + (ap.dot(direction)*direction)/(direction.norm()*direction.norm());
        result.push_back(pproject);
    }
    return result;
}

