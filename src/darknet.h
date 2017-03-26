#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>

static const char * VocNames[] = { "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
};
static int VocN = sizeof(VocNames) / sizeof(VocNames[0]);

class Object {
public:
    std::string name;
    int id;
    float confidence;
    cv::Rect_<float> box;
};

class Yolo {
public:
    Yolo() {};
     ~Yolo();
    bool load();
    bool detect(cv::Mat &img);
    bool release();
    bool getActivations(int i);
    
    std::vector<Object> objects;
    
    // NN filters visualization
    std::vector<cv::Mat> layers;
    std::vector<cv::Mat> layers8;
    int layers_n = 0;
    
    std::string cfg;
    std::string weights;
    bool initialized = false;
    float thresh = .24;

    typedef struct image image;
    typedef struct network network;
    typedef struct box box;
    
    image *darknet_image = NULL;
    image *fixed_size_image = NULL;
    network *net = NULL;
    box *boxes = NULL;
    float **probs = NULL;
    int probs_n = 0;
};
