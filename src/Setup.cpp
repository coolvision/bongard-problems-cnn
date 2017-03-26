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
    
    
    gui.loadFromFile("settings.xml");

    yolo.cfg = ofToDataPath("tiny-yolo-voc.cfg", true);
    yolo.weights = ofToDataPath("tiny-yolo-voc.weights", true);
    yolo.load();
    
//    ofSetMinMagFilters(GL_NEAREST, GL_NEAREST);
}
