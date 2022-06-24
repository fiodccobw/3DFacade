//
// Created by fiodccob on 23-03-22.
//

#include "utils.h"

void Box::update(float X, float Y, float w, float h) {
    x = X;
    y = Y;
    width = w;
    height = h;
}


SurfaceMesh * readMesh(const string& name){
    string file_name = "model/"+name;
    SurfaceMesh* mesh = SurfaceMeshIO::load(file_name);
    if (!mesh) {
        LOG(ERROR) << "Error: failed to load model. Please make sure the file exists and format is correct.";
    }else{
        cout<<"success!"<<endl;
    }
    return mesh;
}

void tokenize(std::string const &str, const char delim,
              std::vector<float> &out)
{
    // construct a stream from the string
    std::stringstream ss(str);

    std::string s;
    while (std::getline(ss, s, delim)) {
        out.push_back(stof(s));
    }
}

Mat3<float> readIntrinsic(const string& path, const string& name){
    string filepath = path+name;
    ifstream file;
    file.open(filepath);
    string str;
    vector<string> txt;
    while(getline(file,str)){
        if(!str.empty()){
            txt.push_back(str);
        }
    }
    file.close();
    Mat3<float> m;
    for(int i=6;i<=8;i++){
        string s = txt[i];
        const char delim = ' ';
        vector<float> out;
        tokenize(s, delim, out);
        m[i-6] = out[0];
        m[i-6+3] = out[1];
        m[i-6+6] = out[2];
    }

    return m;
}

Mat<3,4,float> readExtrinsic(const string& path, const string& name){
    string filepath = path+name;
    ifstream file;
    file.open(filepath);
    string str;
    vector<string> txt;
    while(getline(file,str)){
        if(!str.empty()){
            txt.push_back(str);
        }
    }
    file.close();
    Mat<3,4,float> m;
    for(int i = 1;i<=3;i++){
        string s = txt[i];
        const char delim = ' ';
        vector<float> out;
        tokenize(s, delim, out);
        m[i-1] = out[0];
        m[i-1+3] = out[1];
        m[i-1+6] = out[2];
        m[i-1+9] = out[3];
    }
    return m;
}

double max(std::vector<double> list){
    double max_value = -INFINITY;
    for(auto num:list){
        if(num>max_value){
            max_value = num;
        }
    }
    return max_value;
}


float max(std::vector<float> list){
    float max_value = -INFINITY;
    for(auto num:list){
        if(num>max_value){
            max_value = num;
        }
    }
    return max_value;
}


double min(std::vector<double> list){
    double min_value = INFINITY;
    for(auto num:list){
        if(num<min_value){
            min_value = num;
        }
    }
    return min_value;
}


float min(std::vector<float> list){
    float min_value = INFINITY;
    for(auto num:list){
        if(num<min_value){
            min_value = num;
        }
    }
    return min_value;
}

int max_index(std::vector<double> list){
    double max_value = -INFINITY;
    int index = 0;
    for(int i=0;i<list.size();i++){
        if(list[i]>max_value){
            max_value = list[i];
            index = i;
        }
    }
    return index;
}


int min_index(std::vector<double> list){
    double min_value = INFINITY;
    int index = 0;
    for(int i=0;i<list.size();i++){
        if(list[i]<min_value){
            min_value = list[i];
            index = i;
        }
    }
    return index;
}

std::vector<float> solveFunction_2var(float num1x, float num1y, float sum1, float num2x, float num2y, float sum2){
    float num3 = num1x / num2x;
    float val = num1y - num3 * num2y;
    float val2 = sum1 - sum2 * num3;
    float y = val2 / val;
    float x = (sum1 - num1y * y) / num1x;
    std::vector<float> result;
    result.push_back(x);
    result.push_back(y);
    return result;
}


