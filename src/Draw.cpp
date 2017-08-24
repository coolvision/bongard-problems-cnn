#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "ofApp.h"

void LayerVis::init(layer *l) {
    if (!l) return;
    act_side = ceil(sqrt((l->out_c)));
    act_n = l->out_c;
    act_w = l->out_w;
    act_h = l->out_h;
}

void LayerVis::init(LayerVis *l) {
    if (!l) return;
    act_side = l->act_side;
    act_n = l->act_n;
    act_w = l->act_w;
    act_h = l->act_h;
}

void LayerVis::makeActMaps() {
    
    act_maps.resize(act_n);
    
    for (int i = 0; i < act_side; i++) {
        for (int j = 0; j < act_side; j++) {
            int act_i = i * act_side + j;
            if (act_i < act_n) {                
                if (!grid.m.empty()) {
                    
                    grid.m(cv::Rect(j*act_w, i*act_h,
                                    act_w, act_h)).copyTo(act_maps[act_i].m);
                    act_maps[act_i].makeOF();
                }
            }
        }
    }
}

void LayerVis::drawActMaps(ofPoint off, float zoom) {
    
    ofPushStyle();
    ofSetColor(ofColor::black);
    ofDrawBitmapString(name, off.x, off.y - 10);
    ofPopStyle();
    
    for (int i = 0; i < act_side; i++) {
        for (int j = 0; j < act_side; j++) {
            int act_i = i * act_side + j;
            if (act_i < act_maps.size()) {
                if (!act_maps[act_i].of_img.isAllocated()) continue;
                
                float w = act_maps[act_i].of_img.getWidth() / zoom;
                float h = act_maps[act_i].of_img.getHeight() / zoom;
                
                act_maps[act_i].of_img.draw(off + ofPoint(j * w, i * h),
                            w, h);
            }
        }
    }
}

void LayerVis::copyActMapsFrom(LayerVis *l) {
    
    init(l);
    act_maps.resize(l->act_maps.size());
    for (int i = 0; i < act_maps.size(); i++) {
        act_maps[i].copyFrom(l->act_maps[i].m);
    }
}

void VisImage::copyFrom(Mat _m) {
    if (_m.empty()) return;
    _m.copyTo(m);
    makeOF();
}

void VisImage::load(string img_path) {
    of_img.load(img_path);
    m = toCv(of_img);
}

void VisImage::makeOF() {
    toOf(m, of_img);
    of_img.update();
    of_img.getTextureReference().
    setTextureMinMagFilter(GL_NEAREST,GL_NEAREST);
}

void VisImage::draw(ofPoint off, float zoom) {
    if (!of_img.isAllocated()) return;
    of_img.draw(off, of_img.getWidth() / zoom,
                of_img.getHeight() / zoom);
}

bool ImagesSet::draw(int layer_i, ofPoint layer_offset, float layer_zoom) {

    ofSetColor(ofColor::white);
    
    float w = positives_union.grid.of_img.getWidth() / layer_zoom;
    float h = positives_union.grid.of_img.getHeight() / layer_zoom;
    
    ofPoint off = ofPoint(layer_offset.x + 0,
                          layer_offset.y - h - 10);
    
    positives_union.drawActMaps(off, layer_zoom);
    
    off = ofPoint(layer_offset.x + w + 10,
                  layer_offset.y - h - 10);
    
    negatives_union.drawActMaps(off, layer_zoom);
    
    off = ofPoint(layer_offset.x + 2*w + 2*10,
                  layer_offset.y - h - 10);
    positives_intersection.drawActMaps(off, layer_zoom);
    
    off = ofPoint(layer_offset.x + 0,
                  layer_offset.y - (h + 10) * 2);
    
    color_union.drawActMaps(off, layer_zoom);
    
    off = ofPoint(layer_offset.x + w + 10,
                  layer_offset.y - (h + 10) * 2);
    selected_processed.drawActMaps(off, layer_zoom);
    
    off = ofPoint(layer_offset.x + 2*w + 2*10,
                  layer_offset.y - (h + 10) * 2);
    selected_classified.drawActMaps(off, layer_zoom);
    
    for (int i = 0; i < 6; i++) {
        float w = positives_processed[i].grid.of_img.getWidth() / layer_zoom;
        float h = positives_processed[i].grid.of_img.getHeight() / layer_zoom;
        
        ofPoint off = ofPoint(layer_offset.x + 2*w + 2*10,
                              layer_offset.y + i*h + i*10);
        positives_processed[i].drawActMaps(off, layer_zoom);
    }
    
    for (int j = 0; j < 2; j++) {
        
        for (int i = 0; i < 6; i++) {
            
            int idx = j * 6 + i;
            
            if (layer_i >= 0 && layer_i < images[idx].layers.size()) {
                float w = images[idx].layers[layer_i].grid.of_img.getWidth() / layer_zoom;
                float h = images[idx].layers[layer_i].grid.of_img.getHeight() / layer_zoom;
                
                ofPoint off = ofPoint(layer_offset.x + j*w + j*10,
                                      layer_offset.y + i*h + i*10);
                
                images[idx].layers[layer_i].drawActMaps(off, layer_zoom);
            }
        }
    }
}

bool ImagesSet::drawImages(int layer_i, ofPoint offset, float zoom) {
    
    for (int j = 0; j < 2; j++) {
        
        for (int i = 0; i < 6; i++) {
            
            int idx = j * 6 + i;
            
            float w = images[idx].image.of_img.getWidth();
            float h = images[idx].image.of_img.getHeight();
            ofPoint off = ofPoint(offset.x + j*w/zoom, offset.y + i*h/zoom);
            images[idx].image.draw(off, zoom);
        }
    }
}

void ofApp::draw() {

    
    i1.draw(layer_i, ofPoint(layer_offset->x, layer_offset->y), layer_zoom);
    i1.drawImages(layer_i, ofPoint(offset->x, offset->y), zoom);
    
    gui.draw();
    
    ofSetColor(ofColor::black);
    for (int i = 0; i < dn.net->n; i++) {

        stringstream st;
        if (i < layer_key.size()) {
            if (i == 0) {
                st << ">/";
            } else {
                st << layer_key[i] << "/";
            }
        }
        st << i <<
            " out " << dn.net->layers[i].out_h <<
            "x" << dn.net->layers[i].out_w <<
            "x" << dn.net->layers[i].out_c;
        
        if (i == layer_i) {
            ofDrawBitmapStringHighlight(st.str(), ofGetWindowWidth()-170, 20 + 20*i);
        } else {
            ofDrawBitmapString(st.str(), ofGetWindowWidth()-170, 20 + 20*i);
        }
    }
}
