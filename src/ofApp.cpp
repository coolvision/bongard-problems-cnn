#include "ofApp.h"

void ofApp::update() {
    
    if (process) {
        
        cout << "process!" << endl;
        
        process = false;
        
        ofDirectory dir;
        dir.allowExt("png");
        dir.allowExt("jpg");
        dir.listDir(data_path);
        dir.sort();
        
        data_dir_size = dir.size();
        
        if (data_dir_size > 0) {
            
            if (image_i >= data_dir_size) {
                image_i = data_dir_size - 1;
            }
            if (image_i < 0) image_i = 0;
            
            string path = dir.getPath(image_i);
            image.img.load(path);
            image.img_m = toCv(image.img);
            
//            if (patch_size->x > 1 && patch_size->y > 1 &&
//                image_offset->x + patch_size->x < img_m.cols &&
//                image_offset->y + patch_size->y < img_m.rows) {
//                
//                Mat patch_orig(patch_size->x, patch_size->y, img_m.type());
//                img_m(cv::Rect(image_offset->x, image_offset->y,
//                        patch_orig.cols, patch_orig.rows)).copyTo(patch_orig);
//                
//                Mat patch;
//                resize(patch_orig, patch, cv::Size(0, 0),
//                       patch_zoom, patch_zoom);
//                
//                if (patch_offset->x + patch.cols < img_m.cols &&
//                    patch_offset->y + patch.rows < img_m.rows) {
//                    
//                    img_m.setTo(cv::Scalar(0, 0, 0));
//                    
//                    patch.copyTo(img_m(cv::Rect(patch_offset->x, patch_offset->y,
//                                          patch.cols, patch.rows)));
//                }
//            }
            
            image.img.update();
            
            yolo.detect(image.img_m);
            
//            if (image.layer_i < 0) image.layer_i = 0;
//            if (image.layer_i > yolo.layers_n) image.layer_i = yolo.layers_n;
//            
//            cout << "layer_i " << image.layer_i << endl;
//            
            
            image.layer_img.resize(yolo.net->n);

            for (int i = 0; i < yolo.net->n; i++) {

                yolo.getActivations(i, norm_all);

                toOf(yolo.layers8[i], image.layer_img[i]);
                image.layer_img[i].update();
                image.layer_img[i].getTextureReference().
                    setTextureMinMagFilter(GL_NEAREST,GL_NEAREST);
            }
        }
    }
    
//    if (update_layers_vis) {
//
//    }
}

void ofApp::keyPressed(int key) {

    bool ctrl = ofGetKeyPressed(OF_KEY_COMMAND) ||
                ofGetKeyPressed(OF_KEY_CONTROL);
    
    cout << "keyPressed: " << key << endl;
    
    for (int i = 0; i < layer_key.size(); i++) {
        if (key == layer_key[i]) {
            layer_i = i;
            cout << "keyPressed: layer_i " << layer_i << endl;
            //process = true;
            //update_layers_vis = true;
        }
    }
    
    switch (key) {
        case OF_KEY_LEFT:
            image_i--;
            if (image_i < 0) {
                image_i = 0;
            }
            break;
        case OF_KEY_RIGHT:
            image_i++;
            if (image_i >= data_dir_size) {
                image_i = data_dir_size - 1;
            }
            break;
    };

    if (key == 'u' || key == OF_KEY_LEFT || key == OF_KEY_RIGHT) {
        process = true;
        //update_layers_vis = true;
    }

    else if(key == 's'){
        gui.saveToFile("settings.xml");
    }
    else if(key == 'l'){
        gui.loadFromFile("settings.xml");
    }
}

void ofApp::exit() {
    gui.saveToFile("settings.xml");
}

void ofApp::keyReleased(int key) {
}
void ofApp::mouseMoved(int x, int y ) {
}
void ofApp::mouseDragged(int x, int y, int button) {
}
void ofApp::mousePressed(int x, int y, int button) {
}
void ofApp::mouseReleased(int x, int y, int button) {
}
void ofApp::mouseEntered(int x, int y) {
}
void ofApp::mouseExited(int x, int y) {
}
void ofApp::windowResized(int w, int h) {
}
void ofApp::gotMessage(ofMessage msg) {
}
void ofApp::dragEvent(ofDragInfo dragInfo) {
}
