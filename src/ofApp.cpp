#include "ofApp.h"

void ofApp::update() {
    
    if (process) {
        
        cout << "process!" << endl;
        
        process = false;
        
        cout << "data_path " << data_path << endl;
        
        ofDirectory dir;
//        dir.allowExt("png");
//        dir.allowExt("jpg");
        dir.listDir(data_path);
        dir.sort();
        
        data_dir_size = dir.size();
        
        cout << "data_dir_size " << data_dir_size << endl;
        
        if (data_dir_size > 0) {
            
            if (image_i >= data_dir_size) {
                image_i = data_dir_size - 1;
            }
            if (image_i < 0) image_i = 0;
            
            string path = dir.getPath(image_i);
            
            cout << "path " << path << endl;
            
            images.clear();
            images.resize(4);
            
            for (int j = 0; j < 4; j++) {
                
                string img_path;
                if (j > 1) {
                    img_path = path + "/" + ofToString(j+6) + ".png";
                } else {
                    img_path = path + "/" + ofToString(j) + ".png";
                }

                images[j].img.load(img_path);
                
                cout << "load " << img_path << endl;
                
                images[j].img_m = toCv(images[j].img);
                
                images[j].img.update();
                
                yolo.detect(images[j].img_m);
                
                images[j].layer_img.resize(yolo.net->n);
                
                for (int i = 0; i < yolo.net->n; i++) {
                    
                    yolo.getActivations(i, norm_all);
                    
                    toOf(yolo.layers8[i], images[j].layer_img[i]);
                    images[j].layer_img[i].update();
                    images[j].layer_img[i].getTextureReference().
                    setTextureMinMagFilter(GL_NEAREST,GL_NEAREST);
                }
            }

        }
    }
}

void ofApp::keyPressed(int key) {

    bool ctrl = ofGetKeyPressed(OF_KEY_COMMAND) ||
                ofGetKeyPressed(OF_KEY_CONTROL);
    
    cout << "keyPressed: " << key << endl;
    
    for (int i = 0; i < layer_key.size(); i++) {
        if (key == layer_key[i]) {
            layer_i = i;
            cout << "keyPressed: layer_i " << layer_i << endl;
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
