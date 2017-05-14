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
            img.load(path);
            img_m = toCv(img);
            
   
            
            if (patch_size->x > 1 && patch_size->y > 1 &&
                image_offset->x + patch_size->x < img_m.cols &&
                image_offset->y + patch_size->y < img_m.rows) {
                
                Mat patch_orig(patch_size->x, patch_size->y, img_m.type());
                img_m(cv::Rect(image_offset->x, image_offset->y,
                        patch_orig.cols, patch_orig.rows)).copyTo(patch_orig);
                
                Mat patch;
                resize(patch_orig, patch, cv::Size(0, 0),
                       patch_zoom, patch_zoom);
                
                if (patch_offset->x + patch.cols < img_m.cols &&
                    patch_offset->y + patch.rows < img_m.rows) {
                    
                    img_m.setTo(cv::Scalar(0, 0, 0));
                    
                    patch.copyTo(img_m(cv::Rect(patch_offset->x, patch_offset->y,
                                          patch.cols, patch.rows)));
                }
            }
            
            img.update();
            
            
            yolo.detect(img_m);
            if (layer_i < 1) layer_i = 1;
            if (layer_i > yolo.layers_n) layer_i = yolo.layers_n;
            
            cout << "layer_i " << layer_i << endl;
            
            yolo.getActivations(layer_i-1, norm_all);
            toOf(yolo.layers8[layer_i-1], layer_img);
            layer_img.update();
            layer_img.getTextureReference().setTextureMinMagFilter(GL_NEAREST,GL_NEAREST);
            
//            detections.clear();
//            for (int i = 0; i < yolo.objects.size(); i++) {
//                float x = (float)yolo.objects[i].box.x * (float)img_m.cols;
//                float y = (float)yolo.objects[i].box.y * (float)img_m.rows;
//                float w = (float)yolo.objects[i].box.width * (float)img_m.cols;
//                float h = (float)yolo.objects[i].box.height * (float)img_m.rows;
//                detections.push_back(cv::Rect(x-w/2, y-h/2, w, h));
//            }
        }
    }
}

void ofApp::keyPressed(int key) {

    bool ctrl = ofGetKeyPressed(OF_KEY_COMMAND) ||
                ofGetKeyPressed(OF_KEY_CONTROL);
    
    if (key >= '0' && key <= '9') {
        layer_i = key - '0';
        if (layer_i == 0) layer_i = 10;
        if (ctrl) layer_i += 10;
        cout << "keyPressed: layer_i " << layer_i << endl;
        process = true;
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
