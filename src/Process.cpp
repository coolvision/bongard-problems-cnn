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

bool ImagesSet::extractFetures(Darknet *dn, int layer_i, int selected_image) {
    
    this->layer_i = layer_i;
    
    bool union_init = false;
    
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

            if (i == layer_i) {
                
                if (!union_init) {
                    union_init = true;
                    positives_union.grid.m.create(dn->layers8[layer_i].size(),
                                                  CV_8UC1);
                    positives_union.grid.m.setTo(cv::Scalar(0));
                    
                    negatives_union.grid.m.create(dn->layers8[layer_i].size(),
                                                  CV_8UC1);
                    negatives_union.grid.m.setTo(cv::Scalar(0));
                    
                    positives_union.init(&dn->net->layers[layer_i]);
                    negatives_union.init(&dn->net->layers[layer_i]);
                }
                
                if (j == selected_image) continue;
                
                if (j < 6) {
                    bitwise_or(dn->layers_t[i],
                               positives_union.grid.m,
                               positives_union.grid.m);
                } else {
                    bitwise_or(dn->layers_t[i],
                               negatives_union.grid.m,
                               negatives_union.grid.m);
                }
            }
        }
    }
    
    positives_union.grid.makeOF();
    negatives_union.grid.makeOF();
    
    positives_union.makeActMaps();
    negatives_union.makeActMaps();
}

ClassificationRule ImagesSet::findClassificationRule(int selected_image) {

    positives_processed.resize(6);
    
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
    
    for (int j = 0; j < 6; j++) {
        positives_processed[j].act_maps.clear();
        if (j == selected_image) continue;
        ch[0] = images[j].layers[layer_i].grid.m;
        positives_processed[j].init(&images[j].layers[layer_i]);
        cv::merge(ch, positives_processed[j].grid.m);
        positives_processed[j].grid.makeOF();
        positives_processed[j].makeActMaps();
        positives_processed[j].name = ofToString(j);
    }
    
    for (int j = 0; j < 6; j++) {
        if (j == selected_image) continue;
        for (auto &a: positives_processed[j].act_maps) {
            colorDilate(a.m);
            a.makeOF();
        }
    }
    
    bool positives_intersection_init = false;
    for (int j = 0; j < 6; j++) {
        
        if (j == selected_image) continue;
        
        if (!positives_intersection_init) {
            positives_intersection_init = true;
            positives_intersection.init(&positives_processed[j]);
            positives_intersection.grid.m.create(positives_processed[j].grid.m.size(),
                                                 CV_8UC3);
            positives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
            positives_intersection.makeActMaps();
        }
        
        for (int a = 0; a < positives_intersection.act_maps.size(); a++) {
            bitwise_and(positives_processed[j].act_maps[a].m,
                        positives_intersection.act_maps[a].m,
                        positives_intersection.act_maps[a].m);
            positives_intersection.act_maps[a].makeOF();
        }
    }
    for (auto &a: positives_intersection.act_maps) {
        colorDilate(a.m);
        a.makeOF();
    }
    positives_intersection.name = "positives_intersection";
}

void ImagesSet::classifyPixels(int selected_image) {
    
    if (selected_image >= 0 && selected_image < images.size()) {
        if (layer_i >= 0 && layer_i < images[selected_image].layers.size()) {
            
            vector<Mat> ch;
            Mat blank(positives_union.grid.m.size(), CV_8UC1);
            blank.setTo(cv::Scalar(0));
            ch.resize(3);
            
            ch[0] = images[selected_image].layers[layer_i].grid.m;
            ch[1] = blank;
            ch[2] = blank;
            
            selected_processed.init(&images[selected_image].layers[layer_i]);
            cv::merge(ch, selected_processed.grid.m);
            selected_processed.grid.makeOF();
            selected_processed.makeActMaps();
            
            selected_classified.copyActMapsFrom(&selected_processed);
            
            for (int a = 0; a < selected_classified.act_maps.size(); a++) {
                ::classifyPixels(selected_processed.act_maps[a].m,
                               positives_intersection.act_maps[a].m,
                               selected_classified.act_maps[a].m);
                selected_classified.act_maps[a].makeOF();
            }
        }
    }
}

void classifyPixels(Mat &in, Mat &m, Mat &out) {
    if (in.size() != m.size()) return;
    
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            uint8_t *w = in.data + i * in.cols * 3 + j * 3;
            uint8_t *r = m.data + i * m.cols * 3 + j * 3;
            uint8_t *p = out.data + i * in.cols * 3 + j * 3;

            if (w[0] > 0 && r[0] > 0 && r[1] == 0) {
                p[0] = 255;
                p[1] = 0;
                p[2] = 0;
            } else {
                p[0] = 0;
                p[1] = 0;
                p[2] = 0;
            }
        }
    }
}

void colorDilate(Mat &m) {
    
    Mat tmp;
    for (int d = 0; d < 200; d++) {
        
        m.copyTo(tmp);
        
        int n_added = 0;
        
        for (int i = 1; i < m.rows-1; i++) {
            for (int j = 1; j < m.cols-1; j++) {
                uint8_t *p = m.data + i * m.cols * 3 + j * 3;
                uint8_t *t = tmp.data + i * tmp.cols * 3 + j * 3;
                if (p[0] == 0 && p[1] == 0 && p[2] == 0) {
                    
                    int rn = 0;
                    int gn = 0;
                    for (int q = -1; q <= 1; q++) {
                        for (int w = -1; w <= 1; w++) {
                            if (q == w) continue;
                            uint8_t *p1 = m.data + (i+q) * m.cols * 3 + (j+w) * 3;
                            if (p1[0] > 0) rn++;
                            if (p1[1] > 0) gn++;
                        }
                    }
                    if (rn > 0 && gn == 0) {
                        t[0] = 255;
                        n_added++;
                    } else if (rn == 0 && gn > 0) {
                        t[1] = 255;
                        n_added++;
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


void colorDilatePositive(Mat &m) {
    
    Mat tmp;
    for (int d = 0; d < 100; d++) {
        
        m.copyTo(tmp);
        
        int n_added = 0;
        
        for (int i = 1; i < m.rows-1; i++) {
            for (int j = 1; j < m.cols-1; j++) {
                uint8_t *p = m.data + i * m.cols * 3 + j * 3;
                uint8_t *t = tmp.data + i * tmp.cols * 3 + j * 3;
                if (p[0] == 0 && p[1] == 0 && p[2] == 0) {
                    
                    int rn = 0;
                    int gn = 0;
                    for (int q = -1; q <= 1; q++) {
                        for (int w = -1; w <= 1; w++) {
                            if (q == w) continue;
                            uint8_t *p1 = m.data + (i+q) * m.cols * 3 + (j+w) * 3;
                            if (p1[0] > 0) rn++;
                            if (p1[1] > 0) gn++;
                        }
                    }
                    if (rn > 0 && gn == 0) {
                        t[0] = 255;
                        n_added++;
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
