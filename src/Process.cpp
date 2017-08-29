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
        }
    }
}

bool ImagesSet::processLayer(Darknet *dn, int layer_i, int selected_image) {
    
    this->layer_i = layer_i;
    
    if (layer_i < 0 || layer_i > dn->net->n) return false;
    
    positives_union.clear();
    positives_union.resize(12);

    negatives_union.clear();
    negatives_union.resize(12);
    
    color_union.clear();
    color_union.resize(12);
  
    color_union.clear();
    color_union.resize(12);
    
    color_classified.clear();
    color_classified.resize(12);
    
    // compute color union with each image as selected image
    for (int i = 0; i < 12; i++) {
        
        if (i == 5 || i == 11) continue;
        
        positives_union[i].grid.copyFrom(images[i].layers[layer_i].grid.m);
        positives_union[i].init(&images[i].layers[layer_i]);
        positives_union[i].grid.m.setTo(cv::Scalar(0));
        
        negatives_union[i].grid.copyFrom(images[i].layers[layer_i].grid.m);
        negatives_union[i].init(&images[i].layers[layer_i]);
        negatives_union[i].grid.m.setTo(cv::Scalar(0));
        
        for (int j = 0; j < images.size(); j++) {

            //if (j == i) continue;

            if (j < 6) {
                bitwise_or(images[j].layers[layer_i].grid.m,
                           positives_union[i].grid.m,
                           positives_union[i].grid.m);
            } else {
                bitwise_or(images[j].layers[layer_i].grid.m,
                           negatives_union[i].grid.m,
                           negatives_union[i].grid.m);
            }
        }
        
        positives_union[i].grid.makeOF();
        negatives_union[i].grid.makeOF();

        positives_union[i].makeActMaps();
        negatives_union[i].makeActMaps();
        
        vector<Mat> ch;
        Mat blank(positives_union[i].grid.m.size(), CV_8UC1);
        blank.setTo(cv::Scalar(0));
        
        ch.push_back(positives_union[i].grid.m);
        ch.push_back(negatives_union[i].grid.m);
        ch.push_back(blank);
        
        cv::merge(ch, color_union[i].grid.m);
    
        color_union[i].init(&positives_union[i]);
        color_union[i].grid.makeOF();
        color_union[i].makeActMaps();
        
        if (i < 6) {
            ch[0] = images[i].layers[layer_i].grid.m;
            ch[1] = negatives_union[i].grid.m;
            ch[2] = blank;
        } else {
            ch[0] = positives_union[i].grid.m;
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
        
//        for (int a = 0; a < color_classified[i].act_maps.size(); a++) {
//            ::classifyPixels(color_classified[i].act_maps[a].m,
//                           color_union[i].act_maps[a].m,
//                           color_classified[i].act_maps[a].m);
//            color_classified[i].act_maps[a].makeOF();
//        }
    }
    

    positives_intersection.init(&color_classified[0]);
    positives_intersection.grid.m.create(color_classified[0].grid.m.size(),
                                         color_classified[0].grid.m.type());
    positives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
    positives_intersection.makeActMaps();
    positives_intersection.resizeActMaps(2);

    for (int j = 0; j < 5; j++) {
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

    for (int j = 6; j < 11; j++) {
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
    
    
    
//    negatives_intersection.init(&color_classified[0]);
//    negatives_intersection.grid.m.create(color_classified[0].grid.m.size(),
//                                         CV_8UC3);
//    negatives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
//    negatives_intersection.makeActMaps();
//    
//    for (int j = 6; j < 12; j++) {
//        for (int a = 0; a < negatives_intersection.act_maps.size(); a++) {
//            bitwise_and(color_classified[j].act_maps[a].m,
//                        negatives_intersection.act_maps[a].m,
//                        negatives_intersection.act_maps[a].m);
//            negatives_intersection.act_maps[a].makeOF();
//        }
//    }
    
    

}

ClassificationRule ImagesSet::findClassificationRule(int selected_image) {

//    positives_processed.resize(6);


    for (int selected_i = 0; selected_i < 12; selected_i++) {
        
        for (int j = 0; j < 12; j++) {
            if (j == selected_i) continue;
            
            
            
            
            
            
//        positives_processed[j].act_maps.clear();
//        if (j == selected_image) continue;
//        ch[0] = images[j].layers[layer_i].grid.m;
//        positives_processed[j].init(&images[j].layers[layer_i]);
//        cv::merge(ch, positives_processed[j].grid.m);
//        positives_processed[j].grid.makeOF();
//        positives_processed[j].makeActMaps();
//        positives_processed[j].name = ofToString(j);
            
        }
    }
    
    

        
        
        //        if (j == selected_image) continue;
        //
        //        if (!positives_intersection_init) {
        //            positives_intersection_init = true;
        //            positives_intersection.init(&positives_processed[j]);
        //            positives_intersection.grid.m.create(positives_processed[j].grid.m.size(),
        //                                                 CV_8UC3);
        //            positives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
        //            positives_intersection.makeActMaps();
        //        }
        //
        //        for (int a = 0; a < positives_intersection.act_maps.size(); a++) {
        //            bitwise_and(positives_processed[j].act_maps[a].m,
        //                        positives_intersection.act_maps[a].m,
        //                        positives_intersection.act_maps[a].m);
        //            positives_intersection.act_maps[a].makeOF();
        //        }
//    }
    
    
    
    
    
    
//    cv::merge(ch, color_union.grid.m);
//    
//    color_union.init(&positives_union);
//    color_union.grid.makeOF();
//    color_union.makeActMaps();
    
//    for (int j = 0; j < 6; j++) {
//        positives_processed[j].act_maps.clear();
//        if (j == selected_image) continue;
//        ch[0] = images[j].layers[layer_i].grid.m;
//        positives_processed[j].init(&images[j].layers[layer_i]);
//        cv::merge(ch, positives_processed[j].grid.m);
//        positives_processed[j].grid.makeOF();
//        positives_processed[j].makeActMaps();
//        positives_processed[j].name = ofToString(j);
//    }
    
//    for (int j = 0; j < 6; j++) {
//        if (j == selected_image) continue;
//        for (auto &a: positives_processed[j].act_maps) {
//            colorDilate(a.m);
//            a.makeOF();
//        }
//    }

    
//    bool positives_intersection_init = false;
//    for (int j = 0; j < 12; j++) {

        
//        if (j == selected_image) continue;
//        
//        if (!positives_intersection_init) {
//            positives_intersection_init = true;
//            positives_intersection.init(&positives_processed[j]);
//            positives_intersection.grid.m.create(positives_processed[j].grid.m.size(),
//                                                 CV_8UC3);
//            positives_intersection.grid.m.setTo(cv::Scalar(255, 255, 255));
//            positives_intersection.makeActMaps();
//        }
//        
//        for (int a = 0; a < positives_intersection.act_maps.size(); a++) {
//            bitwise_and(positives_processed[j].act_maps[a].m,
//                        positives_intersection.act_maps[a].m,
//                        positives_intersection.act_maps[a].m);
//            positives_intersection.act_maps[a].makeOF();
//        }
//    }
//    for (auto &a: positives_intersection.act_maps) {
//        colorDilate(a.m);
//        a.makeOF();
//    }
//    positives_intersection.name = "positives_intersection";
}

void ImagesSet::classifyPixels(int selected_image) {
    
//    if (selected_image >= 0 && selected_image < images.size()) {
//        if (layer_i >= 0 && layer_i < images[selected_image].layers.size()) {
//            
//            vector<Mat> ch;
//            Mat blank(positives_union.grid.m.size(), CV_8UC1);
//            blank.setTo(cv::Scalar(0));
//            ch.resize(3);
//            
//            ch[0] = images[selected_image].layers[layer_i].grid.m;
//            ch[1] = blank;
//            ch[2] = blank;
//            
//            selected_processed.init(&images[selected_image].layers[layer_i]);
//            cv::merge(ch, selected_processed.grid.m);
//            selected_processed.grid.makeOF();
//            selected_processed.makeActMaps();
//            
//            selected_classified.copyActMapsFrom(&selected_processed);
//            
//            for (int a = 0; a < selected_classified.act_maps.size(); a++) {
//                ::classifyPixels(selected_processed.act_maps[a].m,
//                               positives_intersection.act_maps[a].m,
//                               selected_classified.act_maps[a].m);
//                selected_classified.act_maps[a].makeOF();
//            }
//        }
//    }
}

void classifyPixels(Mat &in, Mat &m, Mat &out) {
    if (in.size() != m.size()) return;
    
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            uint8_t *w = in.data + i * in.cols * 3 + j * 3;
            uint8_t *r = m.data + i * m.cols * 3 + j * 3;
            uint8_t *p = out.data + i * in.cols * 3 + j * 3;

            if (w[0] > 0) {
                p[0] = r[0];
                p[1] = r[1];
                p[2] = r[2];
            } else {
                p[0] = 0;
                p[1] = 0;
                p[2] = 0;
            }
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

//                    else if (rn == 0 && gn > 0) {
//                        t[1] = 255;
//                        n_added++;
//                    }
                }
            }
        }
        
        tmp.copyTo(m);
        
        if (n_added == 0) {
            break;
        }
    }
}


//void colorDilatePositive(Mat &m) {
//    
//    Mat tmp;
//    for (int d = 0; d < 100; d++) {
//        
//        m.copyTo(tmp);
//        
//        int n_added = 0;
//        
//        for (int i = 1; i < m.rows-1; i++) {
//            for (int j = 1; j < m.cols-1; j++) {
//                uint8_t *p = m.data + i * m.cols * 3 + j * 3;
//                uint8_t *t = tmp.data + i * tmp.cols * 3 + j * 3;
//                if (p[0] == 0 && p[1] == 0 && p[2] == 0) {
//                    
//                    int rn = 0;
//                    int gn = 0;
//                    for (int q = -1; q <= 1; q++) {
//                        for (int w = -1; w <= 1; w++) {
//                            if (q == w) continue;
//                            uint8_t *p1 = m.data + (i+q) * m.cols * 3 + (j+w) * 3;
//                            if (p1[0] > 0) rn++;
//                            if (p1[1] > 0) gn++;
//                        }
//                    }
//                    if (rn > 0 && gn == 0) {
//                        t[0] = 255;
//                        n_added++;
//                    }
//                }
//            }
//        }
//        
//        tmp.copyTo(m);
//        
//        if (n_added == 0) {
//            break;
//        }
//    }
//}
