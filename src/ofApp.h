#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxGui.h"

#include "opencv2/opencv.hpp"

#include "darknet.h"

using namespace cv;
using namespace ofxCv;
using namespace std;

void colorDilate(Mat &m, int c_i = 0);
//void colorDilatePositive(Mat &m);
void classifyPixels(Mat &in, Mat &m, Mat &out);

class VisImage {
public:
    Mat m;
    ofImage of_img;
    
    void copyFrom(Mat _m);
    void load(string img_path);
    void makeOF();
    void draw(ofPoint off, float zoom);
};

// filters responses
// for one image for one layer
class LayerVis {
public:
    
    int act_side;
    int act_n;
    int act_w;
    int act_h;
    
    string name;
    
    vector<VisImage> act_maps;
    vector<VisImage> resized_act_maps;
    
    // all fiters for a layer, combined into a grid
    VisImage grid;
    
    void init(layer *l);
    void init(LayerVis *l);
    
    void copyActMapsFrom(LayerVis *l);
    void makeActMaps();
    void drawActMaps(ofPoint off, float zoom);
    
    void drawResizedActMaps(ofPoint off, float zoom);
    void resizeActMaps(int n);
};

// for one image
// filter responses for all layers
class ImageActivations {
public:
    
    VisImage image;

    vector<LayerVis> layers;
};

class ClassificationRule {
public:
    
    cv::Rect activation_region;
    int layer_i;
    int feature_map_i;
    
    // test if image is positive or negative
    bool apply(ImageActivations &img);
};

class ImagesSet {
public:
    
    ImagesSet() {};
    
// images and NN activations
    vector<ImageActivations> images;
    
    bool load(string path);
    bool extractFetures(Darknet *dn);
    bool processLayer(Darknet *dn, int layer_i, int selected_image);
    ClassificationRule findClassificationRule(int selected_image);
    void classifyPixels(int selected_image);
    
    int layers_n = -1;
    int layer_i = -1;
    
// for visualization only
//===================================================
//    vector<LayerVis> positives_processed;
    LayerVis positives_intersection;
    LayerVis negatives_intersection;

    LayerVis positives_union;
    LayerVis negatives_union;
    LayerVis color_union;
    
    vector<LayerVis> ipl; // interections per layer
    
    vector<LayerVis> color_classified;
    
    bool draw(int layer_i, ofPoint layer_offset, float layer_zoom,
              int selected_image);
    bool drawImages(int layer_i, ofPoint offset, float zoom);
//===================================================
};

class ofApp : public ofBaseApp {

    Darknet dn;
    
    ImagesSet i1;

    int layer_i= 10;
    
    vector<unsigned char> layer_key;
    
    ofPoint off;
    
    bool process = true;

    string data_path;
    int image_i;
    int data_dir_size;
    
    ofxPanel gui;

    // image offset
    ofxVec2Slider offset;
    ofxFloatSlider zoom;

    ofxIntSlider selected_image;
    
    ofxVec2Slider layer_offset;
    ofxFloatSlider layer_zoom;

    ofxToggle norm_all;
    ofxToggle threshold;

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
