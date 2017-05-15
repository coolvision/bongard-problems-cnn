#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "ofApp.h"

//namespace darknet {
//extern "C"
//{
//#include "network.h"
//}
//}
//
//using namespace darknet;    
    
void ofApp::draw() {

    ofSetColor(ofColor::white);
    float w = image.img.getWidth();
    float h = image.img.getHeight();
    off.set(offset->x, offset->y);
    if (image.img.isAllocated()) {
        image.img.draw(off, w / zoom, h / zoom);
    }
    
    if (layer_i >= 0 && layer_i < image.layer_img.size()) {
        if (image.layer_img[layer_i].isAllocated()) {
            image.layer_img[layer_i].draw(ofPoint(layer_offset->x, layer_offset->y),
                           image.layer_img[layer_i].getWidth() / layer_zoom,
                           image.layer_img[layer_i].getHeight() / layer_zoom);
        }
    }
    
    gui.draw();
    
    ofSetColor(ofColor::black);
    for (int i = 0; i < yolo.net->n; i++) {

        stringstream st;
        if (i < layer_key.size()) {
            if (i == 0) {
                st << ">/";
            } else {
                st << layer_key[i] << "/";
            }
        }
        st << i <<
            " out " << yolo.net->layers[i].out_h <<
            "x" << yolo.net->layers[i].out_w <<
            "x" << yolo.net->layers[i].out_c;
        
        if (i == layer_i) {
            ofDrawBitmapStringHighlight(st.str(), ofGetWindowWidth()-170, 20 + 20*i);
        } else {
            ofDrawBitmapString(st.str(), ofGetWindowWidth()-170, 20 + 20*i);
        }
    }
    
    
}
