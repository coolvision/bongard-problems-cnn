#include "ofApp.h"

void ofApp::setup() {

    ofDirectory dir;
    dir.open(ofToDataPath("."));
    data_path = dir.getAbsolutePath();

    image_i = 0;
    data_dir_size = 0;
    process = true;

    gui.setup(); // most of the time you don't need a name
    gui.add(offset.setup("offset", ofPoint(0.0f, 0.0f),
                              ofPoint(-500.0f, -500.0f),
                              ofPoint(500.0f, 500.0f)));
    gui.add(zoom.setup("zoom", 1.0f, 0.1f, 5.0f));

    
    gui.add(layer_offset.setup("layer_offset", ofPoint(0.0f, 0.0f),
                         ofPoint(-500.0f, -500.0f),
                         ofPoint(500.0f, 500.0f)));
    gui.add(layer_zoom.setup("layer_zoom", 1.0f, 0.05f, 4.0f));
    
    gui.add(norm_all.setup("norm_all", false));
    
    
//    gui.add(image_offset.setup("image_offset",
//                               ofPoint(0.0f, 0.0f),
//                               ofPoint(0.0f, 0.0f),
//                               ofPoint(500.0f, 500.0f)));
//    gui.add(patch_size.setup("patch_size",
//                             ofPoint(0.0f, 0.0f),
//                             ofPoint(0.0f, 0.0f),
//                             ofPoint(500.0f, 500.0f)));
//    gui.add(patch_offset.setup("patch_offset",
//                               ofPoint(0.0f, 0.0f),
//                               ofPoint(0.0f, 0.0f),
//                               ofPoint(500.0f, 500.0f)));
//    gui.add(patch_zoom.setup("patch_zoom", 1.0f, 0.5f, 4.0f));
    
    gui.loadFromFile("settings.xml");

    yolo.cfg = ofToDataPath("tiny-yolo-voc.cfg", true);
    yolo.weights = ofToDataPath("tiny-yolo-voc.weights", true);
    
   // yolo.cfg = ofToDataPath("tiny-cars-light.cfg", true);
   // yolo.weights = ofToDataPath("tiny-cars-light_40000.weights", true);
    
    //yolo.cfg = ofToDataPath("shapes_test.cfg", true);
    //yolo.weights = ofToDataPath("shapes_test_160.weights", true);
    
    yolo.cfg = ofToDataPath("shapes_test_7-1.cfg", true);
    yolo.weights = ofToDataPath("shapes_test_7-1_50.weights", true);
    
    yolo.load();

    layer_key.push_back(167);
    layer_key.push_back('1');
    layer_key.push_back('2');
    layer_key.push_back('3');
    layer_key.push_back('4');
    layer_key.push_back('5');
    layer_key.push_back('6');
    layer_key.push_back('7');
    layer_key.push_back('8');
    layer_key.push_back('9');
    layer_key.push_back('0');
    layer_key.push_back('-');
    layer_key.push_back('=');

    
//    ofSetMinMagFilters(GL_NEAREST, GL_NEAREST);
}
