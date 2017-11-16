//
//  ProcessBP.cpp
//  darknet-vis
//
//  Created by sk on 01/07/2017.
//
//

#include "ofApp.h"

bool ImagesSet::load(string path) {
  
    images.clear();
    images.resize(12);

    for (int j = 0; j < 12; j++) {
        
        string img_path;
        img_path = path + "/" + ofToString(j) + ".png";
        
        images[j].image.of_img.load(img_path);
        
        cout << "load " << img_path << endl;
        
        images[j].image.m = toCv(images[j].image.of_img);
    }
}

bool ImagesSet::extractFetures(Darknet *dn) {
    
    for (int j = 0; j < images.size(); j++) {
        
        dn->detect(images[j].image.m);
        images[j].layers.resize(dn->net->n);
        
        layers_n = dn->net->n;
        
        for (int i = 0; i < dn->net->n; i++) {
            
            dn->getActivations(i, true);
            images[j].layers[i].grid.copyFrom(dn->layers_t[i]);
            
            images[j].layers[i].init(&dn->net->layers[i]);
            images[j].layers[i].makeActMaps();

            images[j].layers[i].name = ofToString(j);
        }
    }
}

bool ImagesSet::processLayer(Darknet *dn, int layer_i, int selected_image) {
    
    this->layer_i = layer_i;
    
    if (layer_i < 0 || layer_i > dn->net->n) return false;
    if (images.empty()) return false;
    if (images[0].layers[layer_i].act_n == 0) return false;
        
    color_classified.clear();
    color_classified.resize(12);
    
    positives_union.grid.copyFrom(images[0].layers[layer_i].grid.m);
    positives_union.init(&images[0].layers[layer_i]);
    positives_union.grid.m.setTo(cv::Scalar(0));
    
    negatives_union.grid.copyFrom(images[0].layers[layer_i].grid.m);
    negatives_union.init(&images[0].layers[layer_i]);
    negatives_union.grid.m.setTo(cv::Scalar(0));
    
    for (int j = 0; j < images.size(); j++) {
        
        if (j == selected_image) continue;
        
        if (j < 6) {
            bitwise_or(images[j].layers[layer_i].grid.m,
                       positives_union.grid.m,
                       positives_union.grid.m);
        } else {
            bitwise_or(images[j].layers[layer_i].grid.m,
                       negatives_union.grid.m,
                       negatives_union.grid.m);
        }
    }
    
    positives_union.grid.makeOF();
    negatives_union.grid.makeOF();
    
    positives_union.makeActMaps();
    negatives_union.makeActMaps();
    
    vector<Mat> ch;
    Mat blank(positives_union.grid.m.size(), CV_8UC1);
    blank.setTo(cv::Scalar(0));
    
    ch.push_back(positives_union.grid.m);
    ch.push_back(negatives_union.grid.m);
    ch.push_back(blank);
    
    cv::merge(ch, color_union.grid.m);
    
    color_union.init(&positives_union);
    color_union.grid.makeOF();
    color_union.makeActMaps();
    
    
    for (int i = 0; i < 12; i++) {
        
        if (i == selected_image) continue;
        
        if (i < 6) {
            ch[0] = images[i].layers[layer_i].grid.m;
            ch[1] = negatives_union.grid.m;
            ch[2] = blank;
        } else {
            ch[0] = positives_union.grid.m;
            ch[1] = images[i].layers[layer_i].grid.m;
            ch[2] = blank;
        }
        
        color_classified[i].init(&images[selected_image].layers[layer_i]);
        cv::merge(ch, color_classified[i].grid.m);
        color_classified[i].grid.makeOF();
        color_classified[i].makeActMaps();
        
        color_classified[i].resizeActMaps(2);
        
        if (i < 6) {
            for (auto &a: color_classified[i].resized_act_maps) {
                colorDilate(a.m, 0);
                a.makeOF();
            }
        } else {
            for (auto &a: color_classified[i].resized_act_maps) {
                colorDilate(a.m, 1);
                a.makeOF();
            }
        }
    }

    positives_intersection.init(&color_classified[0]);
    positives_intersection.grid.m.create(color_classified[0].grid.m.size(),
                                         color_classified[0].grid.m.type());
    positives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
    positives_intersection.makeActMaps();
    positives_intersection.resizeActMaps(2);

    for (int j = 0; j < 6; j++) {
        if (j == selected_image) continue;
        for (int a = 0; a < positives_intersection.resized_act_maps.size(); a++) {
            bitwise_and(color_classified[j].resized_act_maps[a].m,
                        positives_intersection.resized_act_maps[a].m,
                        positives_intersection.resized_act_maps[a].m);
            positives_intersection.resized_act_maps[a].makeOF();
        }
    }
    
    for (auto &a: positives_intersection.resized_act_maps) {
        colorDilate(a.m, 0);
        a.makeOF();
    }

    negatives_intersection.init(&color_classified[0]);
    negatives_intersection.grid.m.create(color_classified[0].grid.m.size(),
                                         color_classified[0].grid.m.type());
    negatives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
    negatives_intersection.makeActMaps();
    negatives_intersection.resizeActMaps(2);

    for (int j = 6; j < 12; j++) {
        if (j == selected_image) continue;
        for (int a = 0; a < negatives_intersection.resized_act_maps.size(); a++) {
            bitwise_and(color_classified[j].resized_act_maps[a].m,
                        negatives_intersection.resized_act_maps[a].m,
                        negatives_intersection.resized_act_maps[a].m);
            negatives_intersection.resized_act_maps[a].makeOF();
        }
    }
    
    for (auto &a: negatives_intersection.resized_act_maps) {
        colorDilate(a.m, 1);
        a.makeOF();
    }
}

void colorDilate(Mat &m, int c_i) {
    
    Mat tmp;
    for (int d = 0; d < 10; d++) {
        
        m.copyTo(tmp);
        
        int n_added = 0;
        
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                uint8_t *p = m.data + i * m.cols * 3 + j * 3;
                uint8_t *t = tmp.data + i * tmp.cols * 3 + j * 3;
                if (p[0] == 0 && p[1] == 0 && p[2] == 0) {
                    
                    int rn = 0;
                    int gn = 0;
                    for (int q = -1; q <= 1; q++) {
                        for (int w = -1; w <= 1; w++) {
                            if (q == w) continue;
                            if (i+q < 0 || j+w < 0 ||
                                i+q >= m.rows || j+w >= m.cols) continue;
                            uint8_t *p1 = m.data + (i+q) * m.cols * 3 + (j+w) * 3;
                            if (p1[0] > 0 && p1[1] == 0) rn++;
                            if (p1[1] > 0 && p1[0] == 0) gn++;
                        }
                    }
                    if (c_i == 0) {
                        if (rn > 0) {
                            t[0] = 255;
                            n_added++;
                        }
                    } else {
                        if (gn > 0) {
                            t[0] = 255;
                            n_added++;
                        }
                    }
                }
            }
        }
        
        tmp.copyTo(m);
        
        if (n_added == 0) {
            break;
        }
    }
}

