#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxGui.h"

#include "opencv2/opencv.hpp"

#include "darknet.h"

using namespace cv;
using namespace ofxCv;
using namespace std;

class ofApp : public ofBaseApp {

    Yolo yolo;
    vector<cv::Rect> detections;
    
    ofImage img;
    Mat img_m;
    
    int layer_i = 0;
    
    ofPoint off;
    
    bool process;
    string data_path;
    int image_i;
    int data_dir_size;
    
    ofxPanel gui;

    // image offset
    ofxVec2Slider offset;
    // image scale
    ofxFloatSlider zoom;

public:
    void setup();
    void update();
    void draw();
    void exit();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

};
