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
    cout << "net->n " << net->n << endl;
    for (int i = 0; i < net->n; i++) {
        cout << "layer " << i <<
            " i " << net->layers[i].inputs <<
            " o " << net->layers[i].outputs <<
            " in " << net->layers[i].h <<
            " x " << net->layers[i].w <<
            " x " << net->layers[i].c <<
            " out " << net->layers[i].out_h <<
            " x " << net->layers[i].out_w <<
            " x " << net->layers[i].out_c <<
            " size " << net->layers[i].size <<
            " stride " << net->layers[i].stride <<
            " pad " << net->layers[i].pad << endl;
    }
    
    layers.resize(net->n);
    layers8.resize(net->n);
    layers_n = net->n;
    
    return true;
}

bool Yolo::getActivations(int layer_i) {
    
    // put filter responces from the first layer into an opencv matrix
    if (layer_i >= net->n) return false;
    
    cout << "getActivations " << layer_i << endl;
    
    int act_n = ceil(sqrt((float)net->layers[layer_i].out_c));
    
    cout << "act_n " << act_n << ", total " << net->layers[layer_i].out_c << endl;
    
    // number of output images
    // ok, get one image for a start
    layers[layer_i].create(net->layers[layer_i].out_h * act_n,
                     net->layers[layer_i].out_w * act_n,
                     CV_32F);
    layers8[layer_i].create(net->layers[layer_i].out_h * act_n,
                      net->layers[layer_i].out_w * act_n,
                      CV_8UC1);
    layers8[layer_i].setTo(cv::Scalar(0));
    

    // number of filters
    cv::Mat one_activation(net->layers[layer_i].out_h,
                           net->layers[layer_i].out_w,
                           CV_32F);
    cv::Mat one_activation8(net->layers[layer_i].out_h,
                           net->layers[layer_i].out_w,
                           CV_8UC1);
    
    int act_w = net->layers[layer_i].out_w;
    int act_h = net->layers[layer_i].out_h;
    int act_size = net->layers[layer_i].out_h * net->layers[layer_i].out_w;
    int act_i = 0;
    for (int i = 0; i < act_n; i++) {
        for (int j = 0; j < act_n; j++) {
            act_i = i * act_n + j;
            if (act_i > net->layers[layer_i].out_c) break;
            
            //cout << "set " << i << " " << j << " " << act_i << endl;
            
            memcpy(one_activation.data,
                   net->layers[layer_i].output + act_i * act_size,
                   sizeof(float) * act_size);
            
           // one_activation.convertTo(one_activation8, CV_8UC1, 10.0f);
            
            cv::normalize(one_activation, one_activation8, 0, 255,
                          cv::NORM_MINMAX, CV_8UC1);
        
//            cout << "layers8 " << layers8[i].cols << " " << layers8[i].rows << endl;
//            cout << "copy " << j*act_w << " " << i*act_h << " " << act_w << " " << act_h << endl;
            
            one_activation8.copyTo(layers8[layer_i](cv::Rect(j*act_w, i*act_h,
                                                       act_w, act_h)));
        }
    }
}

    
