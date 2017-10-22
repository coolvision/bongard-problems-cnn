#include "darknet.h"

//using namespace darknet;
using namespace std;

bool Darknet::load() {

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

Darknet::~Darknet() {
    release();
}

bool Darknet::release() {

    free(boxes);
    if (probs) free_ptrs((void **)probs, probs_n);
    if (net) free_network(*net);
    free(net);
}

bool Darknet::detect(cv::Mat &img) {
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

   // *fixed_size_image = resize_image(*darknet_image, net->w, net->h);

    layer l = net->layers[net->n-1];

    if (!fixed_size_image) {
        cout << "!fixed_size_image" << endl;
        return false;
    }
    float *X = darknet_image->data;
    if (!X) {
        cout << "!X" << endl;
        return false;
    }

    clock_t time=clock();
    network_predict(*net, X);
    clock_t time2=clock();
    printf("Predicted in %f seconds. %ld %ld %ld\n", sec(time2-time), CLOCKS_PER_SEC, time, time2);

    // get layers number and info about each layer
    //layers
    cout << "net->n " << net->n << endl;
    for (int i = 0; i < net->n; i++) {
        cout << "layer " << i <<
            " n " << net->layers[i].n <<
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
    layers_t.resize(net->n);

    filters.resize(net->n);
    filters8.resize(net->n);
    filters_t.resize(net->n);

    act_side.resize(net->n);
    act_n.resize(net->n);

    layers_n = net->n;

    return true;
}

bool Darknet::getActivations(int layer_i, bool norm_all) {

    // put filter responces from the first layer into an opencv matrix
    if (layer_i >= net->n) return false;

    cout << "getActivations " << layer_i << " " << net->layers[layer_i].type << endl;

    act_side[layer_i] = ceil(sqrt((float)net->layers[layer_i].out_c));
    act_n[layer_i] = net->layers[layer_i].out_c;
    cout << "act_side " << act_side[layer_i] << ", total " << net->layers[layer_i].out_c << endl;

    // number of output images
    // ok, get one image for a start
    layers[layer_i].create(net->layers[layer_i].out_h * act_side[layer_i],
                     net->layers[layer_i].out_w * act_side[layer_i],
                     CV_32F);
    layers8[layer_i].create(net->layers[layer_i].out_h * act_side[layer_i],
                      net->layers[layer_i].out_w * act_side[layer_i],
                      CV_8UC1);
    layers_t[layer_i].create(net->layers[layer_i].out_h * act_side[layer_i],
                            net->layers[layer_i].out_w * act_side[layer_i],
                            CV_8UC1);

    layers[layer_i].setTo(cv::Scalar(0));
    layers8[layer_i].setTo(cv::Scalar(0));
    layers_t[layer_i].setTo(cv::Scalar(0));

    // number of filters
    cv::Mat one_activation(net->layers[layer_i].out_h,
                           net->layers[layer_i].out_w,
                           CV_32F);
    cv::Mat one_activation8(net->layers[layer_i].out_h,
                           net->layers[layer_i].out_w,
                           CV_8UC1);

    int act_h = net->layers[layer_i].out_h;
    int act_size = net->layers[layer_i].out_h * net->layers[layer_i].out_w;
    int act_i = 0;
    int act_w = net->layers[layer_i].out_w;

    filters[layer_i].resize(act_n[layer_i]);
    filters8[layer_i].resize(act_n[layer_i]);
    filters_t[layer_i].resize(act_n[layer_i]);

    for (int i = 0; i < act_side[layer_i]; i++) {
        for (int j = 0; j < act_side[layer_i]; j++) {
            act_i = i * act_side[layer_i] + j;
            if (act_i >= net->layers[layer_i].out_c) break;

            memcpy(one_activation.data,
                   net->layers[layer_i].output + act_i * act_size,
                   sizeof(float) * act_size);

            one_activation.copyTo(layers[layer_i](cv::Rect(j*act_w, i*act_h,
                                                               act_w, act_h)));
        }
    }
    cv::normalize(layers[layer_i], layers8[layer_i], 0, 255,
                              cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(layers8[layer_i], layers_t[layer_i], 80, 255,
                  cv::THRESH_BINARY);

    return true;
}
