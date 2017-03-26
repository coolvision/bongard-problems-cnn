#include "darknet.h"

extern "C"
{
    #include "detection_layer.h"
    #include "parser.h"
    #include "region_layer.h"
    #include "utils.h"
    extern image ipl_to_image(IplImage* src);
}

using namespace std;

bool Yolo::load() {

    ifstream cfg_stream(cfg.c_str());
    ifstream weights_stream(weights.c_str());
    if (!cfg_stream.good()) {
        cout << "!ccfg_stream.good() " << cfg << endl;
        return false;
    }
    if (!weights_stream.good()) {
        cout << "!weights_stream.good() " << weights << endl;
        return false;
    }
    
    net = (network *)calloc(1, sizeof(network));
    *net = parse_network_cfg((char *)cfg.c_str());

    load_weights(net, (char *)weights.c_str());
    
    set_batch_network(net, 1);
    srand(2222222);
    
    layer l = net->layers[net->n-1];
    
    probs_n = l.w*l.h*l.n;
    boxes = (box *)calloc(probs_n, sizeof(box));
    probs = (float **)calloc(probs_n, sizeof(float *));
    for (int j = 0; j < probs_n; ++j) {
        probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
    }

    initialized = true;

    return true;
}

Yolo::~Yolo() {
    release();
}

bool Yolo::release() {

    free(boxes);
    if (probs) free_ptrs((void **)probs, probs_n);
    if (net) free_network(*net);
    free(net);
}

bool Yolo::detect(cv::Mat &img) {
    if (!initialized) return false;
    
    if (img.empty()) {
        cout << "img.empty()" << endl;
        return false;
    }
    
    if (darknet_image) free_image(*darknet_image);
    if (fixed_size_image) free_image(*fixed_size_image);
    free(darknet_image);
    free(fixed_size_image);
    
    darknet_image = (image *)calloc(1, sizeof(image));
    fixed_size_image = (image *)calloc(1, sizeof(image));
    
    IplImage ipl = img.operator IplImage();
    
    cout << ipl.width << " " << ipl.height << endl;
    *darknet_image = ipl_to_image(&ipl);
    
    *fixed_size_image = resize_image(*darknet_image, net->w, net->h);
    
    layer l = net->layers[net->n-1];
    
    if (!fixed_size_image) {
        cout << "!fixed_size_image" << endl;
        return false;
    }
    float *X = fixed_size_image->data;
    if (!X) {
        cout << "!X" << endl;
        return false;
    }
    
    clock_t time=clock();
    network_predict(*net, X);
    clock_t time2=clock();
    printf("Predicted in %f seconds. %ld %ld %ld\n", sec(time2-time), CLOCKS_PER_SEC, time, time2);

    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, .5);

//    if (l.softmax_tree) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, .4);
//    else do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, .4);

    int num = l.w*l.h*l.n;
    int classes = l.classes;
    
    objects.clear();
    
    int w = ipl.width;
    int h = ipl.height;
    for (int i = 0; i < num; ++i) {
        int class_id = max_index(probs[i], classes);
        float prob = probs[i][class_id];
        if(prob > thresh){
    
            box b = boxes[i];
    
            int left  = (b.x-b.w/2.)*w;
            int right = (b.x+b.w/2.)*w;
            int top   = (b.y-b.h/2.)*h;
            int bot   = (b.y+b.h/2.)*h;
    
            if(left < 0) left = 0;
            if(right > w-1) right = w-1;
            if(top < 0) top = 0;
            if(bot > h-1) bot = h-1;
    
            Object obj;
            obj.box.width = b.w * (float)ipl.width;
            obj.box.height = b.h * (float)ipl.height;
            obj.box.x = b.x * (float)ipl.width - obj.box.width/2;
            obj.box.y = b.y * (float)ipl.height - obj.box.height/2;
            obj.id = class_id;
            objects.push_back(obj);
        }
    }

    // get layers number and info about each layer
    //layers
    
    
    
    
    
    return true;
}

bool Yolo::getActivations(int i) {
    
    // put filter responces from the first layer into an opencv matrix
    
    
    
    
    
}

    
