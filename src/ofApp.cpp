#include "ofApp.h"

void ofApp::update() {
    
    if (process) {
        
        cout << "process!" << endl;
        
        process = false;
        
        cout << "data_path " << data_path << endl;
        
        ofDirectory dir;
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
            
            i1.load(path);
            i1.extractFetures(&dn, layer_i, selected_image);
            i1.processLayer(&dn, layer_i, selected_image);
            
//            i1.findClassificationRule(selected_image);
//            i1.classifyPixels(selected_image);
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
            i1.processLayer(&dn, layer_i, selected_image);
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
