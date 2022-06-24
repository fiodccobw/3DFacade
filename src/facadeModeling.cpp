//
// Created by fiodccob on 23-03-22.
//

#include "facadeModeling.h"

using namespace cv;
using namespace Eigen;


int cmp(const pair<int, float>& x, const pair<int, int>& y)
{
    return x.second > y.second;
}

int cmp2(const pair<int, float>& x, const pair<int, int>& y)
{
    return x.second < y.second;
}

void sortMapByValue(map<int, float>& tMap,vector<pair<int, float> >& tVector)
{
    for (map<int, float>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
        tVector.push_back(make_pair(curr->first, curr->second));

    sort(tVector.begin(), tVector.end(), cmp);
}

void sortMapByValue2(map<int, float>& tMap,vector<pair<int, float> >& tVector)
{
    for (map<int, float>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
        tVector.push_back(make_pair(curr->first, curr->second));

    sort(tVector.begin(), tVector.end(), cmp2);
}

vector<vector<Box>> convert2box(vector<vector<pair<int, float>>> & tVector, vector<Box> boxlist){
    vector<vector<Box>> result;
    for(auto i:tVector){
        vector<Box> newboxlist;
        for(auto j:i){
            Box newbox = boxlist[j.first];
            newboxlist.push_back(newbox);
        }
        result.push_back(newboxlist);
    }
    return result;
}

cv::Scalar rgb_color()
{
    int a = rand()%256,b=rand()%256,c=rand()%256;
    return cv::Scalar(a,b,c);
}
int randomnum()
{
    int a= rand()%1000;
    return a;
}

vector<vector<Box>> alignment(const vector<Box> boxlist,string type, float thre){
    if(type == "midx"){
        map<int, float> midxMap;
        for (int i = 0; i < boxlist.size(); i++) {
            midxMap.insert(make_pair(i, boxlist[i].midx));
        }
        float midx_threshold = thre;
        vector<pair<int, float>> midxVector;
        sortMapByValue(midxMap, midxVector);
        vector<vector<pair<int, float>>> midxgroups;
        vector<pair<int, float>> newgroup;
        for (int i = 0; i < midxVector.size(); i++) {
            if (i == 0) {
                newgroup.push_back(midxVector[i]);
                continue;
            }
            float diff = midxVector[i - 1].second - midxVector[i].second;
            if (diff <= midx_threshold) {
                newgroup.push_back(midxVector[i]);
            } else {
                midxgroups.push_back(newgroup);
                newgroup.clear();
                newgroup.push_back(midxVector[i]);
            }
            if (i == midxVector.size() - 1)
                midxgroups.push_back(newgroup);
        }
        vector<vector<Box>> midxalignment = convert2box(midxgroups, boxlist);
        return midxalignment;
    }else if(type == "midy"){
        map<int, float> midyMap;
        for (int i = 0; i < boxlist.size(); i++) {
            midyMap.insert(make_pair(i, boxlist[i].midy));
        }
        float midy_threshold = thre;
        vector<pair<int, float>> midyVector;
        sortMapByValue(midyMap, midyVector);
        vector<vector<pair<int, float>>> midygroups;
        vector<pair<int, float>> newgroup2;
        for (int i = 0; i < midyVector.size(); i++) {
            if (i == 0) {
                newgroup2.push_back(midyVector[i]);
                continue;
            }
            float diff = midyVector[i - 1].second - midyVector[i].second;
            if (diff <= midy_threshold) {
                newgroup2.push_back(midyVector[i]);
            } else {
                midygroups.push_back(newgroup2);
                newgroup2.clear();
                newgroup2.push_back(midyVector[i]);
            }
            if (i == midyVector.size() - 1)
                midygroups.push_back(newgroup2);
        }
        vector<vector<Box>> midyalignment = convert2box(midygroups, boxlist);
        return midyalignment;
    }else{
        cout<<"please input a correct alignment type!"<<endl;
        vector<vector<Box>> midyalignment;
        return midyalignment;
    }
}




bool checkIntersection(Pt p, Box b){
    if((p.x<=(b.x+b.width))&&(p.x>=b.x)&&(p.y<=(b.y+b.height))&&(p.y>=b.y)){
        return true;
    }else{
        return false;
    }
}

bool checkIntersection_box(Box a, Box b){
    return!((a.x+ a.width) <= b.x ||
            (a.y+a.height) <= b.y ||
            a.x >= (b.x+b.width) ||
            a.y >= (b.y+b.height));
}

MatrixXd randCent(MatrixXd data, int k)
{

    int n = data.cols();
    MatrixXd centroids = MatrixXd::Zero(k, n);
    double min, range;
    for (int i = 0; i < n; i++)
    {
        min = data.col(i).minCoeff();
        range = data.col(i).maxCoeff() - min;
        centroids.col(i) = min * MatrixXd::Ones(k, 1) + MatrixXd::Random(k, 1).cwiseAbs() * range;
    }
    return centroids;
}

double distance(MatrixXd vecA, MatrixXd vecB)
{
    return ((vecA - vecB) * (vecA - vecB).transpose())(0,0);
}

Eigen::MatrixXd kmeans(Eigen::MatrixXd data, int k, Eigen::MatrixXd& centroids){ //k:number of clusters
    int m = data.rows();
    int n = data.cols();
    MatrixXd subCenter = MatrixXd::Zero(m, 2);
    bool change = true;
    while (change)
    {
        change = false;
        for (int i = 0; i < m; i++)
        {
            double minDist = DBL_MAX;
            int minIndex = 0;
            for (int j = 0; j < k; j++)
            {
                double  dist = distance(data.row(i), centroids.row(j));
                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = j;
                }
            }
            if (subCenter(i, 0) != minIndex)
            {
                change = true;
                subCenter(i, 0) = minIndex;
            }
            subCenter(i, 1) = minDist;
        }

        for (int i = 0; i < k; i++)
        {
            MatrixXd sum_all = MatrixXd::Zero(1, n);
            int r = 0;
            for (int j = 0; j < m; j++)
            {
                if (subCenter(j, 0) == i)
                {
                    sum_all += data.row(j);
                    r ++;
                }
            }
            centroids.row(i) = sum_all.row(0) / r;
        }
    }
    return subCenter;
}



cv::Mat facadeModeling::recification(cv::Mat img, string imgname,float ratio, vector<cv::Point> newfourcorner) {

    size_t lastindex = imgname.find_last_of(".");
    string rawname = imgname.substr(0, lastindex);
    bool flag = true;
    std::vector<cv::Point2f> points;
    cv::namedWindow("window", cv::WINDOW_FULLSCREEN);
    for(int i = 0;i<newfourcorner.size();i++){
        if(newfourcorner[i].x<0||newfourcorner[i].y<0){
            flag = false;
            break;
        }
    }
//    flag = false;
    if(flag){
//        cout<<"four points are inside the image!!!!!!!!!!!!!!!!!! No need to select four points!!!!!!!!!"<<endl;
        std::vector<cv::Point2f> temp;
        for(auto p:newfourcorner){
            temp.push_back(cv::Point2f(p.x,p.y));
        }
        int sum = INFINITY;
        int curidx = 0;
        int idx = 0;
        for(auto p:newfourcorner){
            int cursum = p.x+p.y;
            if(sum > cursum){
                sum = cursum;
                idx = curidx;
            }
            curidx++;
        }
        switch(idx){
            case 0:
                points.push_back(temp[0]);
                points.push_back(temp[1]);
                points.push_back(temp[2]);
                points.push_back(temp[3]);
                break;
            case 1:
                points.push_back(temp[1]);
                points.push_back(temp[2]);
                points.push_back(temp[3]);
                points.push_back(temp[0]);
                break;
            case 2:
                points.push_back(temp[2]);
                points.push_back(temp[3]);
                points.push_back(temp[0]);
                points.push_back(temp[1]);
                break;
            case 3:
                points.push_back(temp[3]);
                points.push_back(temp[0]);
                points.push_back(temp[1]);
                points.push_back(temp[2]);
                break;
        }


    }else{
        cv::setMouseCallback("window",click_event,(void*)&points);
        double scale_x = 0.4;
        double scale_y = 0.4;
        cv::Mat scaled;
        cv::resize(img,scaled,cv::Size(),scale_x,scale_y,cv::INTER_LINEAR);

        while(1)
        {
            cv::imshow("window", scaled);
            if(points.size() == 4){
                for (auto & point : points)
                {
                    cout<<"X and Y coordinates are given below"<<endl;
                    cout<<point.x<<'\t'<<point.y<<endl;
                }

                break;
            }

            cv::waitKey(10);
        }
    }



    //Compute quad point for edge

    // compute the size of the card by keeping aspect ratio.
//    float ratio=1.6;
    //Or you can give your own height
    float cardW=sqrt((points[3].x-points[0].x)*(points[3].x-points[0].x)+(points[3].y-points[0].y)*(points[3].y-points[0].y));
    float cardH=cardW/ratio;
    cv::Rect R(points[0].x,points[0].y,cardW,cardH);

    Point R1=Point2f(R.x,R.y);
    Point R2=Point2f(R.x,R.y+R.height);
    Point R3=Point2f(Point2f(R.x+R.width,R.y+R.height));
    Point R4=Point2f(Point2f(R.x+R.width,R.y));

//    std::vector<Point2f> quad_pts;
    std::vector<Point2f> squre_pts;

//    quad_pts.push_back(Q1);
//    quad_pts.push_back(Q2);
//    quad_pts.push_back(Q3);
//    quad_pts.push_back(Q4);

    squre_pts.push_back(R1);
    squre_pts.push_back(R2);
    squre_pts.push_back(R3);
    squre_pts.push_back(R4);


    cv::Mat transmtx = getPerspectiveTransform(points,squre_pts);
    int offsetSize=150;
    cv::Mat transformed = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    warpPerspective(img, transformed, transmtx, transformed.size());

    //rectangle(src, R, Scalar(0,255,0),1,8,0);


    imwrite("rectified_img/"+imgname,transformed);


    cv::FileStorage fs("rectified/"+rawname+".yml", FileStorage::WRITE);
    fs<<"transform mat"<<transmtx;
    fs.release();

//    imshow("src",src);
//    waitKey();


    return transformed;
}

void facadeModeling::click_event(int event, int x, int y, int flags, void *param) {
    if(event == cv::EVENT_LBUTTONDOWN) {
        std::vector<cv::Point2f>* ptPtr = (std::vector<cv::Point2f>*)param;
        ptPtr->push_back(cv::Point2f(2.5*x,2.5*y));
    }
}


vector<cv::Vec4i> facadeModeling::edgedetection(cv::Mat img, vector<Box> windowlist,int i) {
    cv::Mat image = cv::imread("rectified_img/20220427_145119.jpg");
    float width = float(image.size().width);
    float height = float(image.size().height);
    cv::Mat gus;
//    cv::fastNlMeansDenoising(img, gus, 20, 7, 21);
    cv::Mat grayImage;
    cvtColor(image,grayImage,COLOR_BGR2GRAY);
    cv::Mat detected;
    cv::Canny(grayImage,detected,120,360);
    double scale_x = 0.4;
    double scale_y = 0.4;
    cv::Mat scaled;


//    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);
//    double start = double(getTickCount());
    vector<cv::Vec4i> vecLines;
//    double times = (double(getTickCount()) - start) * 1000 / getTickFrequency();
//    std::cout<<"times:" <<times<<"ms"<<std::endl;
//    cv::Mat reslineMat;
//    lsd->detect(grayImage,vecLines);
//    cv::Mat lines = img;
//    for( size_t i = 0; i < vecLines.size(); i++ )
//    {
//        int length = (vecLines[i][1] - vecLines[i][3])*(vecLines[i][1] - vecLines[i][3]) + (vecLines[i][0] - vecLines[i][2])*(vecLines[i][0] - vecLines[i][2]);
//        int k;
//        if((vecLines[i][0] - vecLines[i][2]) == 0){
//            k = INFINITY;
//        }else{
//            k = abs((vecLines[i][1] - vecLines[i][3])/(vecLines[i][0] - vecLines[i][2]));
//        }
//        cv::Point mid((vecLines[i][0] + vecLines[i][2])/2,(vecLines[i][1] + vecLines[i][3])/2);


//        if(((abs(vecLines[i][0] - vecLines[i][2])<10)||(abs(vecLines[i][1] - vecLines[i][3])<10))&&length > 400)
//        if(k>4||k<0.25){
//            line( img, Point(vecLines[i][0], vecLines[i][1]),
//                  Point(vecLines[i][2], vecLines[i][3]), Scalar(255,0,255), 2, 8 );
//        }


//        if(length > 5000){
//            for(auto b:windowlist){
//                if(b.visited == 0){
//                    //vertical 1
//                    if(k>2){
//                        float dist1 = abs(float(mid.x) - b.x);
//                        float dist2 = abs(float(mid.x) - (b.x+b.width));
//                        if((dist1 <= 10 || dist2<=10)&&float(mid.y)<=(b.y+b.height) && float(mid.y)>= b.y){
//                            line( img, Point(vecLines[i][0], vecLines[i][1]),
//                                  Point(vecLines[i][2], vecLines[i][3]), Scalar(0,255,255), 3, 8 );
//                        }
//                    }else if(k<0.5){
//                        float dist1 = abs(float(mid.y) - b.y);
//                        float dist2 = abs(float(mid.y) - (b.y+b.height));
//                        if((dist1 <= 10 || dist2<=10)&&float(mid.x)<=(b.x+b.width) && float(mid.x)>= b.x){
//                            line( img, Point(vecLines[i][0], vecLines[i][1]),
//                                  Point(vecLines[i][2], vecLines[i][3]), Scalar(0,255,255), 3, 8 );
//                        }
//                    }
//                }
//
//            }
//        }
//    }
//    for( size_t i = 0; i < vecLines.size(); i++ ){
//        line( img, Point(vecLines[i][0], vecLines[i][1]),
//              Point(vecLines[i][2], vecLines[i][3]), Scalar(0,0,255), 3, 8 );
//    }
//    cv::resize(img,scaled,cv::Size(),scale_x,scale_y,cv::INTER_LINEAR);
//    cv::imshow("reslineMat",scaled);
    cv::imwrite("rectified/r"+ to_string(20) +".jpg",detected);
//    cv::imwrite("/home/fiodccob/GEO2020/regularization/layout/yolov5/data/samples2/r"+ to_string() +".jpg",img);

//    cv::waitKey(0);
    return vecLines;
}

//int edgetest(Box b, vector<cv::Vec4i> vecLines){
//    float totaldist1=0; //vertical 1
//    float totaldist2=0; //vertical 2
//    float totaldist3=0; //horizontal 1
//    float totaldist4=0; //horizontal 2
//    vector<cv::Vec4i> list1;
//    vector<cv::Vec4i> list2;
//    vector<cv::Vec4i> list3;
//    vector<cv::Vec4i> list4;
//    for( size_t i = 0; i < vecLines.size(); i++ )
//    {
//        int length = (vecLines[i][1] - vecLines[i][3])*(vecLines[i][1] - vecLines[i][3]) + (vecLines[i][0] - vecLines[i][2])*(vecLines[i][0] - vecLines[i][2]);
//        int k;
//        if((vecLines[i][0] - vecLines[i][2]) == 0){
//            k = INFINITY;
//        }else{
//            k = abs((vecLines[i][1] - vecLines[i][3])/(vecLines[i][0] - vecLines[i][2]));
//        }
//        cv::Point mid((vecLines[i][0] + vecLines[i][2])/2,(vecLines[i][1] + vecLines[i][3])/2);
//        if(length>5000){
//
//        }
//    }
//}


vector<Box> getGrid(vector<Box> windowlist, float midy_threshold, float midx_threshold, int& boxsize){
    vector<Box> newwindowlist;
    vector<float> verticallines;
    vector<float> horizontallines;
    vector<vector<Box>> midx = alignment(windowlist,"midx",midy_threshold);

    for(const auto& i:midx){
        float sum=0;
        for(auto ii:i){
            sum+=ii.x+ii.width/2;
        }
        float avg = sum/float(i.size());
        verticallines.push_back(avg);
    }



    //1.1.2 horizontal lines
    vector<vector<Box>> midy = alignment(windowlist,"midy",midx_threshold);

    for(const auto& i:midy){
        float sum=0;
        for(auto ii:i){
            sum+=ii.y+ii.height/2;
        }
        float avg = sum/float(i.size());
        horizontallines.push_back(avg);
    }



//    cout<<"length:"<<toeliminate.size()<<endl;

    for(auto i:windowlist){
        bool flag = true;
//        for(auto ii:list1){
//            if(i.id == ii){
//                flag = true;
//            }
//        }
//        for(auto ii:list2){
//            if(i.id == ii){
//                flag = true;
//            }
//        }
//        for(auto ii:toeliminate){
//            if(i.id == ii){
//                flag = false;
//            }
//        }
        if(flag)
            newwindowlist.push_back(i);
    }

    //1.1.4 add the missing boxes
    //vertical lines again on the updated boxlist
    midx = alignment(newwindowlist,"midx",midx_threshold);
    verticallines.clear();



//    vector<float> columnwidth;
    for(const auto& i:midx){
        float sum=0,sum2 = 0;
        for(auto ii:i){
            sum+=ii.x+ii.width/2;
//            sum2+=ii.width;
        }
        float avg = sum/float(i.size()),avg2 = sum2/float(i.size());
        verticallines.push_back(avg);
//        columnwidth.push_back(avg2);
    }
    //horizontal lines 2
    midy = alignment(newwindowlist,"midy",midy_threshold);
    horizontallines.clear();
    for(const auto& i:midy){
        float sum=0, sum2 = 0, sum3 =0;
        for(auto ii:i){
            sum+=ii.y+ii.height/2;
            sum2+=ii.height;
            sum3+=ii.width;
        }
        float avg = sum/float(i.size()), avg2 = sum2/float(i.size()),avg3 = sum3/float(i.size());
        horizontallines.push_back(avg);
//        rowheight.push_back(avg2);
//        columnwidth.push_back(avg3);
    }
        float columnwidth;
        float rowheight;
        int count = 0;
        float sumwid = 0;
        float sumheight = 0;
        for(auto b:windowlist){
            sumwid+=b.width;
            sumheight+=b.height;
            count++;
        }
        columnwidth = sumwid/float(count);
        rowheight = sumheight/float(count);
//        cout<<"wid: "<<columnwidth<<" height: "<<rowheight<<"--------------------------"<<endl;
    //update the boxlist
//    for(auto i:list2){
//        list1.push_back(i);
//    }
//    for(auto i:list1){
//        bool flag = true;
//        for(auto ii:toeliminate){
//            if(i == ii) {
//                flag = false;
//                break;
//            }
//        }
//        if(flag){
//            newwindowlist.push_back(boxlist[i]);
//        }
//    }
//    cout<<"wid: "<<horizontallines.size()<<" height: "<<verticallines.size()<<"--------------------------"<<endl;
    //add boxes
    vector<Pt> positions;
    vector<pair<float,float>> positionsize;
    vector<Box> boxestoadd;
//    int curid = boxsize;
    for(int i=0;i<verticallines.size();i++){
        for(int ii=0;ii<horizontallines.size();ii++){
            Pt newpoint(verticallines[i],horizontallines[ii]);
            positions.push_back(newpoint);
//            positionsize.push_back(make_pair(columnwidth,rowheight));
        }
    }
    for(int i=0;i<positions.size();i++){
        bool flag=true;
        Box newbox(boxsize,positions[i].x-columnwidth/2,positions[i].y-rowheight/2,columnwidth,rowheight,0);
//        for(int ii=0;ii<tempboxlist.size();ii++){
//            if(checkIntersection(positions[i],tempboxlist[ii])){
//                flag = false;
//                break;
//            }
//            if(checkIntersection_box(newbox,tempboxlist[ii])){
//                flag = false;
//                break;
//            }
//        }
        for(auto ii:newwindowlist){
            if(checkIntersection_box(newbox,ii)){
                flag = false;
                break;
            }
        }
        if(flag){
            newwindowlist.push_back(newbox);
            boxsize++;
        }

    }
//    for(auto p:newwindowlist){
//        cout<<"boxx: "<<p.id<<endl;
//    }
    return newwindowlist;
}

vector<Box> facadeModeling::regularization(string imagename, int mode, int fix, vector<cv::Point2f> fpimg, int numf) {

    //parameters
    bool roof = true;
    float midx_threshold = 40;
    float midy_threshold = 40;
    int minimum_point = 1;
    float epsilon = 0.04;
    float spacing_threshold = 6;

    vector<Box> boxlist;
    float width, height;
    cv::Mat image;
    cv::Mat image2;
    if(mode == 0){
        size_t lastindex = imagename.find_last_of(".");
        string rawname = imagename.substr(0, lastindex);
        cout<<"name: "<<imagename<<endl;
        string image_name = "rectified_img/"+imagename;
        string json_name = "objectdetection/predictions/"+rawname+".json";

//        string txt_name = "yolov5/runs/detect/exp/labels/"+rawname+".txt";


        //read image

        image = cv::imread(image_name);
        image2 = image.clone();
//        image2 = cv::imread(image_name);
        width = float(image.size().width);
        height = float(image.size().height);

        //read json
        std::ifstream jfile(json_name);
        nlohmann::json j = nlohmann::json::parse(jfile);
        auto blist = j["boxes"].get<vector<vector<int>>>();
        auto typelist = j["labels"].get<vector<int>>();
        //create boxlist

        for (int i = 0; i < blist.size(); i++) {
            float x1 = blist[i][0], y1 = blist[i][1], x2 = blist[i][2], y2 = blist[i][3];
            float box_w = x2 - x1;
            float box_h = y2 - y1;

            Box b(i, x1, y1, box_w, box_h, typelist[i]);
            boxlist.push_back(b);
        }
//        cout<<"size!"<<boxlist.size()<<endl;
    }else if(mode == 1){
        size_t lastindex = imagename.find_last_of(".");
        string rawname = imagename.substr(0, lastindex);
        string image_name = "yolov5/data/samples2/"+imagename;
//        string image_name = "yolov5/runs/detect/exp2/cmp_x0058.jpg";
        string txt_name = "yolov5/runs/detect/exp/labels/"+rawname+".txt";
//        string txt_name = "yolov5/runs/detect/exp2/labels/cmp_x0058.txt";
        image = cv::imread(image_name);



        width = float(image.size().width);
        height = float(image.size().height);
//        cout<<"width: "<<width<<" height: "<<height<<endl;
        std::ifstream file(txt_name);
//        cout<<"string: "<<txt_name<<endl;
        string str;
        int index = 0;
        while (std::getline(file, str))
        {

            vector<std::string> list;
            std::istringstream iss(str);
            for (std::string s; iss >> s; )
                list.push_back(s);
            float x = stof(list[1])*width;
            float y = stof(list[2])*height;
            float w = stof(list[3])*width;
            float h = stof(list[4])*height;
//            cout<<"type: "<<list[0]<<endl;
//            cout<<"width"<<w<<endl;
            Box newb(index,x-w/2,y-h/2,w,h, stoi(list[0]));
            boxlist.push_back(newb);
            index++;
//                cout<<"string: "<<s<<endl;
        }
    }
//    cout<<"test: "<<boxlist.size()<<endl;
    if(boxlist.empty()){
//        cout<<"!"<<endl;
        return boxlist;
    }

    vector<Box> windowlist;
    vector<Box> doorlist;
    vector<Box> balconylist;

    //extend of the face
    vector<float> xlist,ylist;
    for(auto p:fpimg){
        xlist.push_back(p.x);
        ylist.push_back(p.y);
    }
    float minx = min(xlist);
    float maxx = max(xlist);
    float miny = min(ylist);
    float maxy = max(ylist);
    vector<Box> tempboxlist;
    for(auto i:boxlist){

        if(i.x>=minx && i.x<=maxx && i.y>=miny && i.y<=maxy && i.x+i.width>=minx && i.x+i.width<=maxx && i.y+i.height>=miny && i.y+i.height<=maxy ){
            tempboxlist.push_back(i);
        }else if(i.y>=miny && i.y<=maxy && i.y+i.height>maxy){
            tempboxlist.push_back(i);
        }
        else{
            continue;
        }

    }

    for(auto i:tempboxlist){
        if(i.type == 0){
            windowlist.push_back(i);
        }else if(i.type == 2){
            doorlist.push_back(i);
        }else if(i.type == 1){
            balconylist.push_back(i);
        }
    }

    vector<Box> finallist;

    if(windowlist.empty() || windowlist.size()==1){
        return windowlist;
    }


    //edges
//    edgedetection(image,windowlist,1);
    //

    //1 preprocessing
    //1.1 eliminate outliers and add missing


    vector<vector<Box>> midxalignment;
    vector<vector<Box>> midyalignment;
    vector<Box> newwindowlist;
    vector<float> verticallines;
    vector<float> horizontallines;




    if(fix){
        //1.1.1 vertical lines
        vector<vector<Box>> midx = alignment(windowlist,"midx",midy_threshold);

        for(const auto& i:midx){
            float sum=0;
            for(auto ii:i){
                sum+=ii.x+ii.width/2;
            }
            float avg = sum/float(i.size());
            verticallines.push_back(avg);
        }



        //1.1.2 horizontal lines
        vector<vector<Box>> midy = alignment(windowlist,"midy",midx_threshold);

        for(const auto& i:midy){
            float sum=0;
            for(auto ii:i){
                sum+=ii.y+ii.height/2;
            }
            float avg = sum/float(i.size());
            horizontallines.push_back(avg);
        }


        //1.1.3 eliminate the outliers and semi-outliers, store the semi-outliers
        vector<int> list1,list2;
        for(auto i:midx){
            if(i.size()==1){
                list1.push_back(i[0].id);
            }
        }
        for(auto i:midy){
            if(i.size()==1){
                list2.push_back(i[0].id);
            }
        }
        vector<int> toeliminate;
        for(auto i:list1){
            for(auto ii:list2){
                if(i==ii)
                    toeliminate.push_back(i);
            }
        }

        for(auto i:windowlist){
            bool flag = true;
            for(auto ii:list1){
                if(i.id == ii){
                    flag = true;
                }
            }
            for(auto ii:list2){
                if(i.id == ii){
                    flag = true;
                }
            }
        for(auto ii:toeliminate){
            if(i.id == ii){
                flag = false;
            }
        }
            if(flag)
                newwindowlist.push_back(i);
        }

        //1.1.4 add the missing boxes
        //vertical lines again on the updated boxlist
    midx = alignment(newwindowlist,"midx",midx_threshold);
    verticallines.clear();
//    vector<float> columnwidth;
    for(const auto& i:midx){
        float sum=0,sum2 = 0;
        for(auto ii:i){
            sum+=ii.x+ii.width/2;
//            sum2+=ii.width;
        }
        float avg = sum/float(i.size()),avg2 = sum2/float(i.size());
        verticallines.push_back(avg);
//        columnwidth.push_back(avg2);
    }
    //horizontal lines 2
    midy = alignment(newwindowlist,"midy",midy_threshold);
    horizontallines.clear();
    vector<float> rowheight;
    vector<float> columnwidth;
    for(const auto& i:midy){
        float sum=0, sum2 = 0, sum3 =0;
        for(auto ii:i){
            sum+=ii.y+ii.height/2;
            sum2+=ii.height;
            sum3+=ii.width;
        }
        float avg = sum/float(i.size()), avg2 = sum2/float(i.size()),avg3 = sum3/float(i.size());
        horizontallines.push_back(avg);
        rowheight.push_back(avg2);
        columnwidth.push_back(avg3);
    }

    //update the boxlist

    //add boxes
    vector<Pt> positions;
    vector<pair<float,float>> positionsize;
    vector<Box> boxestoadd;
    int curid = boxlist.size();
    for(int i=0;i<verticallines.size();i++){
        for(int ii=0;ii<horizontallines.size();ii++){
            Pt newpoint(verticallines[i],horizontallines[ii]);
            positions.push_back(newpoint);
            positionsize.push_back(make_pair(columnwidth[ii],rowheight[ii]));
        }
    }
    for(int i=0;i<positions.size();i++){
        bool flag=true;
        Box newbox(curid,positions[i].x-positionsize[i].first/2,positions[i].y-positionsize[i].second/2,positionsize[i].first,positionsize[i].second,0);
        for(int ii=0;ii<tempboxlist.size();ii++){
            if(checkIntersection(positions[i],tempboxlist[ii])){
                flag = false;
                break;
            }
            if(checkIntersection_box(newbox,tempboxlist[ii])){
                flag = false;
                break;
            }
        }
        for(auto ii:newwindowlist){
            if(checkIntersection_box(newbox,ii)){
                flag = false;
                break;
            }
        }
        if(flag){
            newwindowlist.push_back(newbox);
            curid++;
        }

    }


        //midx
        midxalignment = alignment(newwindowlist,"midx",midx_threshold);
        //midy
        midyalignment = alignment(newwindowlist,"midy",midy_threshold);

    }else{
        newwindowlist = windowlist;
        //midx
        midxalignment = alignment(windowlist,"midx",midx_threshold);
        //midy
        midyalignment = alignment(windowlist,"midy",midy_threshold);
    }
    cout<<"boxlistsize: "<<newwindowlist.size()<<endl;

    int a = randomnum();

    image = cv::Scalar(255,255,255);
    for(auto i:tempboxlist){
//        cout<<"x: "<<i.x<<" y: "<<i.y<<endl;
//        cout<<"width: "<<i.width<<" height: "<<i.height<<endl;
        cv::rectangle(image,cv::Point(i.x,i.y),cv::Point(i.x+i.width,i.y+i.height),cv::Scalar(0,0,255),10);
    }

    //kmeans
    vector<cv::Point2f> samples;
    for(int i=0;i<newwindowlist.size();i++) {
        cv::Point2f temp(newwindowlist[i].width, newwindowlist[i].height);
        samples.push_back(temp);
    }
    vector<int> labels;
    cv::kmeans(samples,numf,labels,TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),10,KMEANS_PP_CENTERS);
    vector<vector<Box>> samesize2;
    for(int i=0;i<numf;i++){
        vector<Box> newb;
        samesize2.push_back(newb);
    }

    for(int i=0;i<newwindowlist.size();i++){
        samesize2[labels[i]].push_back(newwindowlist[i]);
    }


    vector<Box> nnnn = newwindowlist;
    //2. quadratic programming
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    int N = newwindowlist.size();
    cout<<"cur size: "<<N<<endl;
    //variable declarartions
    GRBVar x[N],y[N],w[N],h[N];
    map<int,int> id2idx;
    for(int i=0; i < N; i++){
        x[i] = model.addVar(0.0,width,newwindowlist[i].x,GRB_CONTINUOUS);
        y[i] = model.addVar(0.0,height,newwindowlist[i].y,GRB_CONTINUOUS);
        w[i] = model.addVar(0.0,3*newwindowlist[i].width,newwindowlist[i].width,GRB_CONTINUOUS);
        h[i] = model.addVar(0.0,3*newwindowlist[i].height,newwindowlist[i].height,GRB_CONTINUOUS);
        id2idx.insert(make_pair(newwindowlist[i].id,i));
    }

    //update
    model.update();


    //constraint declaration
    for(int i=0;i<midxalignment.size();i++){
        for(int ii=0;ii<midxalignment[i].size()-1;ii++){
            GRBLinExpr lhs;
            lhs = x[id2idx[midxalignment[i][ii].id]] + 0.5*w[id2idx[midxalignment[i][ii].id]] - (x[id2idx[midxalignment[i][ii+1].id]] + 0.5*w[id2idx[midxalignment[i][ii+1].id]]);
            model.addConstr(lhs == 0);
        }

    }

    for(int i=0;i<midyalignment.size();i++){
        for(int ii=0;ii<midyalignment[i].size()-1;ii++){
            GRBLinExpr lhs;
            lhs = y[id2idx[midyalignment[i][ii].id]] + 0.5*h[id2idx[midyalignment[i][ii].id]] - (y[id2idx[midyalignment[i][ii+1].id]] + 0.5*h[id2idx[midyalignment[i][ii+1].id]]);
            model.addConstr(lhs == 0);
        }

    }

    for(auto i:samesize2){
        if(i.size()>1){
            for(int ii = 0; ii<i.size()-1;ii++){
                GRBLinExpr lhs1, lhs2;
                lhs1 = w[id2idx[i[ii].id]] - w[id2idx[i[ii+1].id]];
                lhs2 = h[id2idx[i[ii].id]] - h[id2idx[i[ii+1].id]];
                model.addConstr(lhs1 == 0);
                model.addConstr(lhs2 == 0);
            }
        }
    }



    //objective function
    GRBQuadExpr obj = 0;
    float weight = 2.5;
    for(int i=0;i<N;i++){
        obj+=(x[i]+0.5*w[i]-newwindowlist[i].x-0.5*newwindowlist[i].width)*(x[i]+0.5*w[i]-newwindowlist[i].x-0.5*newwindowlist[i].width)+
             (y[i]+0.5*h[i]-newwindowlist[i].y-0.5*newwindowlist[i].height)*(y[i]+0.5*h[i]-newwindowlist[i].y-0.5*newwindowlist[i].height)+
             weight*((w[i]-newwindowlist[i].width)*(w[i]-newwindowlist[i].width)+(h[i]-newwindowlist[i].height)*(h[i]-newwindowlist[i].height));

    }
    model.setObjective(obj);

    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        for(int i=0;i<N;i++){
            newwindowlist[i].update(x[i].get(GRB_DoubleAttr_X),y[i].get(GRB_DoubleAttr_X),w[i].get(GRB_DoubleAttr_X),h[i].get(GRB_DoubleAttr_X));
        }
    }else if (status == GRB_INF_OR_UNBD) {
        cout << "Infeasible or unbounded" << endl;
    } else if (status == GRB_INFEASIBLE) {
        cout << "Infeasible" << endl;
    } else if(status == GRB_UNBOUNDED) {
        cout << "Uunbounded" << endl;
    } else {
        cout << "Optimization was stopped with status" << status << endl;
    }

    image = cv::Scalar(255,255,255);
    for(auto i:newwindowlist){
        cv::rectangle(image,cv::Point(i.x,i.y),cv::Point(i.x+i.width,i.y+i.height),cv::Scalar(0,0,255),10);
    }
    cv::imwrite("detectionresults/"+ imagename,image);


    //handel the balcony


    if(!balconylist.empty()){

        for(int i=0;i<newwindowlist.size();i++){
            for(auto bb:balconylist){
                if(checkIntersection_box(bb,newwindowlist[i])){
                    newwindowlist[i].update(newwindowlist[i].x,newwindowlist[i].y,newwindowlist[i].width,bb.y-newwindowlist[i].y-1);
                }
            }
        }


        for(auto i:balconylist){
            newwindowlist.push_back(i);
        }
    }

    for(int i=0;i<newwindowlist.size();i++){
        if(newwindowlist[i].y>=miny && newwindowlist[i].y<=maxy && newwindowlist[i].y+newwindowlist[i].height>maxy){
            newwindowlist[i].update(newwindowlist[i].x,newwindowlist[i].y,newwindowlist[i].width,maxy-newwindowlist[i].y-1);
        }
    }


    if(!doorlist.empty()){

        for(int i=0;i<doorlist.size();i++){
            if(doorlist[i].y>=miny && doorlist[i].y<=maxy && doorlist[i].y+doorlist[i].height>maxy){
                doorlist[i].update(doorlist[i].x,doorlist[i].y,doorlist[i].width,maxy-doorlist[i].y-1);
            }
            newwindowlist.push_back(doorlist[i]);
        }
    }
    return newwindowlist;
}




facadeModeling::facadeModeling() = default;




