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
            extractFeture(i, j, 0);
            extractFeture(i, j, 1);
            extractFeture(i, j, 2);
            extractFeture(i, j, 3);
            extractFeture(i, j, 4);
        }
        
        if (debug_print) cout << "image " << j << " " << features[j].size() << endl;
    }
    
    if (features.empty()) return false;
    
    bool found = false;
    bool test_correct = false;
    int solution_layer_i = -1;
    int solution_map_i = -1;
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
                if (debug_print) cout << "(valid) classify with positive = 1: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
                found = true;
                solution_layer_i = features[0][i].layer_i;
                solution_map_i = features[0][i].map_i;
                if (features[5][i].state == 1) {
                    if (debug_print) cout << "correct: " << features[5][i].state << " " << features[11][i].state << endl;
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
            if (found) break;
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
                if (debug_print) cout << "(valid) classify with positive = 0: " << features[0][i].i << " " << features[0][i].layer_i << " " << features[0][i].map_i << endl;
                found = true;
                solution_layer_i = features[0][i].layer_i;
                solution_map_i = features[0][i].map_i;
                if (features[5][i].state == 0) {
                    if (debug_print) cout << "correct: " << features[5][i].state << " " << features[11][i].state << endl;
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
            if (found) break;
        }
    }
    
    if (found) {
        solved = true;
        correct = test_correct;
        cout << problem_i << " solved (" << solution_layer_i << "," << solution_map_i << ") " << test_correct << endl;
    }
}

bool ImagesSet::extractFeture(int i, int j, int index) {
    
    for (int a = 0; a < images[j].layers[i].resized_act_maps.size(); a++) {
        
        Mat &m = images[j].layers[i].resized_act_maps[a].m;
        
        Feature f;
        f.i = index;
        f.layer_i = i;
        f.map_i = a;
        f.image_i = j;
        f.state = 0;
        
        if (m.cols == 2 && m.rows == 2) {
            uint8_t *p = m.data;
            if (index == 0) {
                if (p[0] > 0 || p[1] > 0 || p[2] > 0 || p[3] > 0) {
                    f.state = 1;
                }
            } else if (index == 1) {
                if (p[0] > 0 || p[2] > 0) {
                    f.state = 1;
                }
            } else if (index == 2) {
                if (p[1] > 0 || p[3] > 0) {
                    f.state = 1;
                }
            } else if (index == 3) {
                if (p[0] > 0 || p[1] > 0) {
                    f.state = 1;
                }
            } else if (index == 4) {
                if (p[2] > 0 || p[3] > 0) {
                    f.state = 1;
                }
            } else {
                return false;
            }
        }
        
        features[j].push_back(f);
    }
}

bool ImagesSet::processLayer(Darknet *dn, int layer_i, int selected_image) {
    
    this->layer_i = layer_i;
    
    if (selected_image < 0 || selected_image >= 12) return false;
    if (layer_i < 0 || layer_i > dn->net->n) return false;
    if (images.empty()) return false;
    if (images[0].layers[layer_i].act_n == 0) return false;
    
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
    
    positives_union.resizeActMaps(2);
    negatives_union.resizeActMaps(2);
    
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

    color_union.resizeActMaps(2);
    
    ipl[layer_i].init(&images[selected_image].layers[layer_i]);
    ipl[layer_i].copyActMapsFrom(&images[selected_image].layers[layer_i]);
}

