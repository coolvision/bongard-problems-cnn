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

void LayerVis::resizeActMaps(int n) {
    
    resized_act_maps.resize(act_maps.size());
    for (int a = 0; a < act_maps.size(); a++) {
        
        int w = act_maps[a].m.cols / n;
        int h = act_maps[a].m.rows / n;
        
        resized_act_maps[a].m.create(n, n, act_maps[a].m.type());
        resized_act_maps[a].m.setTo(Scalar(0, 0, 0));
        int bpp = resized_act_maps[a].m.channels();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                
                int n_r = 0;
                int n_g = 0;
                int n_b = 0;
                for (int y = i*h; y < i*h+h; y++) {
                    for (int x = j*w; x < j*w+w; x++) {
                        uint8_t *p = act_maps[a].m.data +
                            y * act_maps[a].m.cols * bpp + x * bpp;
                        if (bpp == 1) {
                            if (p[0] > 0) n_r++;
                        } else {
                            if (p[0] > 0) n_r++;
                            if (p[1] > 0) n_g++;
                        }
                    }
                }
                
                uint8_t *p = resized_act_maps[a].m.data +
                    i * n * bpp + j * bpp;
                
                if (bpp == 1) {
                    if (n_r > 0)
                        p[0] = 255;
                } else {
                    if (n_r > 0)
                        p[0] = 255;
                    if (n_g > 0)
                        p[1] = 255;
                }
            }
        }
        resized_act_maps[a].makeOF();
    }
}

void LayerVis::drawResizedActMaps(ofPoint off, float zoom) {
    
    ofPushStyle();
    ofSetColor(ofColor::black);
    ofDrawBitmapString(name, off.x, off.y - 10);
    ofPopStyle();
    
    int margin = 2;
    
    for (int i = 0; i < act_side; i++) {
        for (int j = 0; j < act_side; j++) {
            int act_i = i * act_side + j;
            if (act_i < act_maps.size() && act_i < resized_act_maps.size()) {
                if (!act_maps[act_i].of_img.isAllocated()) continue;
                if (!resized_act_maps[act_i].of_img.isAllocated()) continue;
                
                float w = act_maps[act_i].of_img.getWidth() / zoom;
                float h = act_maps[act_i].of_img.getHeight() / zoom;
                
                resized_act_maps[act_i].of_img.draw(off +
                                ofPoint(j * w + j * margin, i * h + i * margin),
                                w, h);
            }
        }
    }
}

void LayerVis::drawActMaps(ofPoint off, float zoom) {
    
    int off_y = 10;
    
    int margin = 2;
    
    for (int i = 0; i < act_side; i++) {
        for (int j = 0; j < act_side; j++) {
            int act_i = i * act_side + j;
            if (act_i < act_maps.size()) {
                if (!act_maps[act_i].of_img.isAllocated()) continue;
                
                float w = act_maps[act_i].of_img.getWidth() / zoom;
                float h = act_maps[act_i].of_img.getHeight() / zoom;
                act_maps[act_i].of_img.draw(off +
                            ofPoint(j * w + j * margin, i * h + i * margin),
                            w, h);
            }
        }
    }
    
    ofPushStyle();
    ofSetColor(ofColor::black);
    ofDrawBitmapString(name, off.x, off.y - off_y);
    ofPopStyle();
}

void LayerVis::copyActMapsFrom(LayerVis *l) {
    
    init(l);
    act_maps.resize(l->act_maps.size());
    for (int i = 0; i < act_maps.size(); i++) {
        act_maps[i].copyFrom(l->act_maps[i].m);
    }
    
    resized_act_maps.resize(l->resized_act_maps.size());
    for (int i = 0; i < resized_act_maps.size(); i++) {
        resized_act_maps[i].copyFrom(l->resized_act_maps[i].m);
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

bool ImagesSet::draw(int layer_i, ofPoint layer_offset, float layer_zoom,
                     int selected_image) {

    ofSetColor(ofColor::white);

    int margin = 20;
    margin = positives_union.act_w;

    float w = 2 * positives_union.act_side + (positives_union.act_w * positives_union.act_side) / layer_zoom;
    float h = 2 * positives_union.act_side + (positives_union.act_h * positives_union.act_side) / layer_zoom;
    w += margin;
    h += margin;
    
    ofPoint off = ofPoint(layer_offset.x,
                          layer_offset.y);

    positives_union.drawResizedActMaps(off, layer_zoom);

    off.x += w;
    negatives_union.drawResizedActMaps(off, layer_zoom);

    off.x += w;
    color_union.drawResizedActMaps(off, layer_zoom);
    
    
    for (int j = 0; j < 2; j++) {
        
        for (int i = 0; i < 6; i++) {
            
            int idx = j * 6 + i;
            
            if (layer_i >= 0 && layer_i < images[idx].layers.size()) {

                ofPoint off = ofPoint(layer_offset.x + j*w,
                                      layer_offset.y + i*h + h);
                
                images[idx].layers[layer_i].drawActMaps(off, layer_zoom);
            }
        }
    }

    for (int j = 0; j < 2; j++) {
        
        for (int i = 0; i < 6; i++) {
            
            int idx = j * 6 + i;
            
            if (layer_i >= 0 && layer_i < images[idx].layers.size()) {
                
                ofPoint off = ofPoint(layer_offset.x + j*w + w*2 + margin,
                                      layer_offset.y + i*h + h);
                
                images[idx].layers[layer_i].drawResizedActMaps(off, layer_zoom);
            }
        }
    }
    
    int lh = 0;
    for (int i = 0; i < layers_n; i++) {
        
        ofPoint off = ofPoint(layer_offset.x + 2*w + w*2 + margin,
                              layer_offset.y + lh);
        
        ipl[i].drawResizedActMaps(off, layer_zoom);

        lh += 10 + 2 * ipl[i].act_side + (ipl[i].act_h * ipl[i].act_side) / layer_zoom;
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
    
    ofSetBackgroundColor(100, 100, 100, 255);

    i1.draw(layer_i, ofPoint(layer_offset->x, layer_offset->y),
            layer_zoom, selected_image);
    
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
