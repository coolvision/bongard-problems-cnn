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
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            
            int idx = j * 2 + i;
            
            float w = images[idx].img.getWidth();
            float h = images[idx].img.getHeight();
            off.set(offset->x + j*w/zoom, offset->y + i*h/zoom);
            if (images[idx].img.isAllocated()) {
                images[idx].img.draw(off, w / zoom, h / zoom);
            }
            
            if (layer_i >= 0 && layer_i < images[idx].layer_img.size()) {
                float w = images[idx].layer_img[layer_i].getWidth() / layer_zoom;
                float h = images[idx].layer_img[layer_i].getHeight() / layer_zoom;
            
                ofPoint off = ofPoint(layer_offset->x + j*w + j*10,
                                      layer_offset->y + i*h + i*10);
                
                if (images[idx].layer_img[layer_i].isAllocated()) {
                    images[idx].layer_img[layer_i].draw(off, w, h);
                }
            }
            
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
