#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>

namespace darknet {
extern "C"
{
    #include "detection_layer.h"
    #include "parser.h"
    #include "region_layer.h"
    #include "utils.h"
    extern image ipl_to_image(IplImage* src);
}
}

using namespace darknet;

class Object {
public:
    std::string name;
    int id;
    float confidence;
    cv::Rect_<float> box;
};

typedef struct image image;
typedef struct network network;
typedef struct box box;
typedef struct layer layer;

class Darknet {
public:
    Darknet() {};
     ~Darknet();
    bool load();
    bool detect(cv::Mat &img);
    bool release();
    bool getActivations(int i, bool norm_all);
    
    // filters visualization
    // all filters for each layer
    std::vector<std::vector<cv::Mat> > filters;
    std::vector<std::vector<cv::Mat> > filters8;
    std::vector<std::vector<cv::Mat> > filters_t;
   
    // grid of filters for each layer
    std::vector<cv::Mat> layers;
    std::vector<cv::Mat> layers8;
    std::vector<cv::Mat> layers_t;
    
    std::vector<int> act_side;
    std::vector<int> act_n;
    
    int layers_n = 0;
    
    std::string cfg;
    std::string weights;
    bool initialized = false;
    float thresh = .24;

    image *darknet_image = NULL;
    image *fixed_size_image = NULL;
    network *net = NULL;
    box *boxes = NULL;
    float **probs = NULL;
    int probs_n = 0;
};
