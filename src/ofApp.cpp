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
            
            yolo.detect(img_m);
            
            detections.clear();
            cout << "objects " << yolo.objects.size() << endl;
            for (int i = 0; i < yolo.objects.size(); i++) {
                float x = (float)yolo.objects[i].box.x * (float)img_m.cols;
                float y = (float)yolo.objects[i].box.y * (float)img_m.rows;
                float w = (float)yolo.objects[i].box.width * (float)img_m.cols;
                float h = (float)yolo.objects[i].box.height * (float)img_m.rows;
                detections.push_back(cv::Rect(x-w/2, y-h/2, w, h));
                cout << "x " << x << " " << y << " " << w << " " << h << endl;
            }
        }
    }
}

void ofApp::keyPressed(int key) {

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
