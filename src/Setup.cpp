#include "ofApp.h"

void ofApp::setup() {

    ofDirectory dir;
    dir.open(ofToDataPath("./BP"));
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
                         ofPoint(-1600.0f, -1600.0f),
                         ofPoint(1600.0f, 1600.0f)));
    gui.add(layer_zoom.setup("layer_zoom", 1.0f, 0.05f, 4.0f));
    
    //gui.add(norm_all.setup("norm_all", false));
    //gui.add(threshold.setup("threshold", true));
    
    gui.add(selected_image.setup("selected_image", 0, 0, 11));
    
    gui.loadFromFile("settings.xml");

    dn.cfg = ofToDataPath("tiny-dn-voc.cfg", true);
    dn.weights = ofToDataPath("tiny-dn-voc.weights", true);
    
    dn.cfg = ofToDataPath("shapes_test_7-1.cfg", true);
    dn.weights = ofToDataPath("shapes_test_7-1_50.weights", true);
    
//    dn.cfg = ofToDataPath("shapes_test_7-4.cfg", true);
//    dn.weights = ofToDataPath("shapes_test_7-4_11.weights", true);
    
    dn.cfg = ofToDataPath("shapes_test_7-4-26.cfg", true);
    dn.weights = ofToDataPath("shapes_test_7-4-26_8.weights", true);
    
    
    dn.load();

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
}
