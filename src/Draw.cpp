#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "ofApp.h"

void ofApp::draw() {

    ofSetColor(ofColor::white);
    float w = img.getWidth();
    float h = img.getHeight();
    off.set(offset->x, offset->y);
    if (img.isAllocated()) {
        img.draw(off, w / zoom, h / zoom);
    }
    
    if (!yolo.layers8.empty() && layer_img.isAllocated()) {
        layer_img.draw(ofPoint(layer_offset->x, layer_offset->y),
                       layer_img.getWidth() / layer_zoom,
                       layer_img.getHeight() / layer_zoom);
    }
    
    
    for (int i = 0; i < yolo.objects.size(); i++) {
        ofSetColor(ofColor::darkRed);
        ofSetLineWidth(2);
        ofNoFill();
        ofDrawRectangle(off.x + (float)yolo.objects[i].box.x/zoom,
                        off.y + (float)yolo.objects[i].box.y/zoom,
                        (float)yolo.objects[i].box.width/zoom,
                        (float)yolo.objects[i].box.height/zoom);
        ofDrawBitmapStringHighlight(VocNames[yolo.objects[i].id],
                           off.x + (float)yolo.objects[i].box.x/zoom,
                           off.y + (float)yolo.objects[i].box.y/zoom);
        
    }
    
    gui.draw();
}

