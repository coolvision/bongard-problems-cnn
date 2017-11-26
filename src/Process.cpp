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
        
        //cout << "load " << img_path << endl;
        
        images[j].image.m = toCv(images[j].image.of_img);
    }
}

bool ImagesSet::extractFetures(Darknet *dn, int problem_i) {
    
    features.resize(12);
    
    bool debug_print = false;
    bool debug_print1 = false;
    
    for (int j = 0; j < images.size(); j++) {
        
        features[j].clear();
        
        dn->detect(images[j].image.m);
        images[j].layers.resize(dn->net->n);
        
        layers_n = dn->net->n;
        
        for (int i = 0; i < dn->net->n; i++) {
            
            dn->getActivations(i, true);
            images[j].layers[i].grid.copyFrom(dn->layers_t[i]);
            
            images[j].layers[i].init(&dn->net->layers[i]);
            images[j].layers[i].makeActMaps();

            images[j].layers[i].resizeActMaps(2);
            
            images[j].layers[i].name = ofToString(j);
        }
        
        for (int i = 9; i >= 6; i--) {
            for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
                
                Mat &m = images[j].layers[i].resized_act_maps[a].m;
                
                Feature f;
                f.i = 0;
                f.layer_i = i;
                f.map_i = a;
                f.image_i = j;
                f.state = 0;
                
                if (m.cols == 2 && m.rows == 2) {
                    uint8_t *p = m.data;
                    if (p[0] > 0 || p[1] > 0 || p[2] > 0 || p[3] > 0) {
                        f.state = 1;
                    }
                }
                
                features[j].push_back(f);
            }
//        }
//
//        for (int i = 9; i >= 6; i--) {
            for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
                
                Mat &m = images[j].layers[i].resized_act_maps[a].m;
                
                Feature f;
                f.i = 1;
                f.layer_i = i;
                f.map_i = a;
                f.image_i = j;
                f.state = 0;
                
                if (m.cols == 2 && m.rows == 2) {
                    uint8_t *p = m.data;
                    if (p[0] > 0 || p[2] > 0) {
                        f.state = 1;
                    }
                }
                
                features[j].push_back(f);
            }
//        }
//        
//        for (int i = 9; i >= 6; i--) {
            for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
                
                Mat &m = images[j].layers[i].resized_act_maps[a].m;
                
                Feature f;
                f.i = 2;
                f.layer_i = i;
                f.map_i = a;
                f.image_i = j;
                f.state = 0;
                
                if (m.cols == 2 && m.rows == 2) {
                    uint8_t *p = m.data;
                    if (p[1] > 0 || p[3] > 0) {
                        f.state = 1;
                    }
                }
                
                features[j].push_back(f);
            }
//        }
//
//        for (int i = 9; i >= 6; i--) {
            for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
                
                Mat &m = images[j].layers[i].resized_act_maps[a].m;
                
                Feature f;
                f.i = 3;
                f.layer_i = i;
                f.map_i = a;
                f.image_i = j;
                f.state = 0;
                
                if (m.cols == 2 && m.rows == 2) {
                    uint8_t *p = m.data;
                    if (p[0] > 0 || p[1] > 0) {
                        f.state = 1;
                    }
                }
                
                features[j].push_back(f);
            }
//        }
//        
//        for (int i = 9; i >= 6; i--) {
            for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
                
                Mat &m = images[j].layers[i].resized_act_maps[a].m;
                
                Feature f;
                f.i = 4;
                f.layer_i = i;
                f.map_i = a;
                f.image_i = j;
                f.state = 0;
                
                if (m.cols == 2 && m.rows == 2) {
                    uint8_t *p = m.data;
                    if (p[2] > 0 || p[3] > 0) {
                        f.state = 1;
                    }
                }
                
                features[j].push_back(f);
            }
        }
        
        if (debug_print) cout << "image " << j << " " << features[j].size() << endl;
    }
    
    if (features.empty()) return false;
    
    bool found = false;
    bool test_correct = false;
    for (int i = 0; i < features[0].size(); i++) {

        // positive = 1
        bool correct = true;
        for (int j = 0; j < 5; j++) {
            if (features[j][i].state == 0) {
                correct = false;
                break;
            }
        }
        for (int j = 6; j < 11; j++) {
            if (features[j][i].state == 1) {
                correct = false;
                break;
            }
        }
        if (correct) {
            if (features[5][i].state != features[11][i].state) {
                if (debug_print1) cout << "(valid) classify with positive = 1: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
                for (int j = 0; j < 12; j++) {
                    if (debug_print1) cout << features[j][i].state << " ";
                }
                if (debug_print1) cout << endl;
                found = true;
                if (features[5][i].state == 1) {
//                if (features[5][i].state == 1 && features[11][i].state == 0) {
                    if (debug_print1) cout << "correct: " << features[5][i].state << " " << features[11][i].state << endl;
                    test_correct = true;
                } else {
                    test_correct = false;
                }
            } else {
                if (debug_print) cout << "(not valid) classify with positive = 1: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
            }
            for (int j = 0; j < 12; j++) {
                if (debug_print) cout << features[j][i].state << " ";
            }
            if (debug_print) cout << endl;
            if (!debug_print) if (found) break;
        }
        
        // positive = 0
        correct = true;
        for (int j = 0; j < 5; j++) {
            if (features[j][i].state == 1) {
                correct = false;
                break;
            }
        }
        for (int j = 6; j < 11; j++) {
            if (features[j][i].state == 0) {
                correct = false;
                break;
            }
        }
        if (correct) {
            if (features[5][i].state != features[11][i].state) {
//            if (features[5][i].state == 0) {
                if (debug_print1) cout << "(valid) classify with positive = 0: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
                for (int j = 0; j < 12; j++) {
                    if (debug_print1) cout << features[j][i].state << " ";
                }
                if (debug_print1) cout << endl;
                found = true;
                if (features[5][i].state == 0) {
//                if (features[5][i].state == 0 && features[11][i].state == 1) {
                    if (debug_print1) cout << "correct: " << features[5][i].state << " " << features[11][i].state << endl;
                    test_correct = true;
                } else {
                    test_correct = false;
                }
            
            } else {
                if (debug_print) cout << "(not valid) classify with positive = 0: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
            }
            for (int j = 0; j < 12; j++) {
                if (debug_print) cout << features[j][i].state << " ";
            }
            if (debug_print) cout << endl;
            if (!debug_print) if (found) break;
        }
    }
    
    if (found) {
        cout << problem_i << " solved, " << test_correct << endl;
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
        
//        if (i < 6) {
//            for (auto &a: color_classified[i].resized_act_maps) {
//                colorDilate(a.m, 0);
//                a.makeOF();
//            }
//        } else {
//            for (auto &a: color_classified[i].resized_act_maps) {
//                colorDilate(a.m, 1);
//                a.makeOF();
//            }
//        }
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
    
//    for (auto &a: positives_intersection.resized_act_maps) {
//        colorDilate(a.m, 0);
//        a.makeOF();
//    }
    
    ipl[layer_i].init(&positives_intersection);
    ipl[layer_i].copyActMapsFrom(&positives_intersection);
    
    for (int n = 0; n < ipl[layer_i].resized_act_maps.size(); n++) {
        
        Mat &m = ipl[layer_i].resized_act_maps[n].m;
        
        ClassifyingFeature cf;
        cf.layer_i = layer_i;
        cf.map_i = n;
       
        
        int filled_n = 0;
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                uint8_t *p = m.data + i * m.cols * 3 + j * 3;
                cf.map.push_back(*p);
                if (p[0] > 0 && p[1] == 0 && p[2] == 0) {
                    filled_n++;
                    //cout << "map: " << n << "; i: " << i << " j: " << j << endl;
                }
            }
        }
        
        cf.filled_n = filled_n;
        if (filled_n > 0) {
            common_features.push_back(cf);
        }
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

