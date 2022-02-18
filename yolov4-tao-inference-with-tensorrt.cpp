#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "dlfcn.h"


#define DEVICE 0  // GPU id


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)


using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 384;
static const int INPUT_W = 1248;
const char* INPUT_BLOB_NAME = "Input";
//const char* OUTPUT_BLOB_NAME1 = "num_detections";
//const char* OUTPUT_BLOB_NAME2 = "nmsed_boxes";
//const char* OUTPUT_BLOB_NAME3 = "nmsed_scores";
//const char* OUTPUT_BLOB_NAME4 = "nmsed_classes";
const char* OUTPUT_BLOB_NAME1 = "BatchedNMS";
const char* OUTPUT_BLOB_NAME2 = "BatchedNMS_1";
const char* OUTPUT_BLOB_NAME3 = "BatchedNMS_2";
const char* OUTPUT_BLOB_NAME4 = "BatchedNMS_3";
static Logger gLogger;

struct Object{
    int classid;
    float confidence;
    std::string classname;
    float left;
    float top;
    float width;
    float height;
};

struct BindInfo{
    std::string bindname;
    long bindsize;
    int bindindex;
};

std::vector<BindInfo> getBindInfo(ICudaEngine* engine){
    std::vector<BindInfo>binfos;
    for(int i = 0; i< engine->getNbBindings();i ++){
        BindInfo binfo;
        std::cout << "bindindex:" << i << " bindname:" << engine->getBindingName(i) << std::endl;
        binfo.bindname = std::string(engine->getBindingName(i));
        Dims output_dims = engine->getBindingDimensions(i);
        long size = 1;
        for(int j = 1;j < output_dims.nbDims;j ++){
//            std::cout << "dim:" << output_dims.d[j] << std::endl;
            size *= output_dims.d[j];
        }
        binfo.bindsize = size;
        binfo.bindindex = i;
        binfos.push_back(binfo);
    }
    return std::move(binfos);
}

std::vector<Object> NvDsInferParseCustomBatchedNMSTLT (
        int output_0,
        std::vector<std::vector<float>> outputs,
        float threshold,
        int NetworkH,
        int NetworkW,
        int orig_h,
        int orig_w, float ratio = 1.0) {
    std::vector<Object>objectList;
    if(outputs.size() != 3)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 3 output buffers, detected in the network :"
                  << outputs.size() << std::endl;
        return std::move(objectList);
    }
    int p_keep_count = output_0;
    float* p_bboxes = (float *) outputs[0].data();
    float* p_scores = (float *) outputs[1].data();
    float* p_classes = (float *) outputs[2].data();

    const int keep_top_k = 200;

    for (int i = 0; i < p_keep_count && objectList.size() <= keep_top_k; i++) {
        std::cout << p_scores[i] << " vs " << threshold << " index:" << i << std::endl;
        if ( p_scores[i] < threshold) continue;

//        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        Object object;
        object.classid = (int) p_classes[i];
        object.confidence = p_scores[i];
        std::cout << "classid:" << object.classid << " Confidence:" << object.confidence << std::endl;

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(p_bboxes[4*i] * NetworkW, 0, NetworkW - 1);
        object.top = CLIP(p_bboxes[4*i+1] * NetworkH, 0, NetworkH - 1);
        object.width = CLIP(p_bboxes[4*i+2] * NetworkW, 0, NetworkW - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3] * NetworkH, 0, NetworkH - 1) - object.top;
        float ratio_w = float(orig_w) / NetworkW;
        float ratio_h = float(orig_h) / NetworkH;
//        object.left = object.left * ratio_w;
//        object.top = object.top * ratio_h;
//        object.width = object.width * ratio_w;
//        object.height = object.height * ratio_w;

        object.left = object.left * ratio;
        object.top = object.top * ratio;
        object.width = object.width * ratio;
        object.height = object.height * ratio;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return std::move(objectList);
}


void doInference(IExecutionContext& context, float* input, int batchSize, std::vector<BindInfo>& binfos,int& output_0,
                 std::vector<std::vector<float>>& other_outputs) {
    void* buffers[5];
    other_outputs.resize(3);
//    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[0], batchSize * binfos[0].bindsize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[1], batchSize * binfos[1].bindsize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers[2], batchSize * binfos[2].bindsize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[3], batchSize * binfos[3].bindsize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[4], batchSize * binfos[4].bindsize * sizeof(float)));
    other_outputs[0].resize(batchSize * binfos[2].bindsize);
    other_outputs[1].resize(batchSize * binfos[3].bindsize);
    other_outputs[2].resize(batchSize * binfos[4].bindsize);
    Dims inputDims;
    inputDims.nbDims = 4;
    inputDims.d[0] = batchSize;
    inputDims.d[1] = 3;
    inputDims.d[2] = 384;
    inputDims.d[3] = 1248;
    context.setBindingDimensions(0, inputDims);


    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * binfos[0].bindsize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(&output_0, buffers[1], batchSize * binfos[1].bindsize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(other_outputs[0].data(), buffers[2], batchSize * binfos[2].bindsize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(other_outputs[1].data(), buffers[3], batchSize * binfos[3].bindsize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(other_outputs[2].data(), buffers[4], batchSize * binfos[4].bindsize * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    CUDA_CHECK(cudaFree(buffers[2]));
    CUDA_CHECK(cudaFree(buffers[3]));
    CUDA_CHECK(cudaFree(buffers[4]));
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    initLibNvInferPlugins(&gLogger, "");
    char *trtModelStream{nullptr};
    size_t size{0};
    // std::ifstream file("/media/wzq/Seagate/ubuntu1604/zhihuiyan/deepModelTraining/TAO/tools/saved.engine", std::ios::binary);
    std::ifstream file("/media/wzq/Seagate/ubuntu1604/zhihuiyan/deepModelTraining/TAO/tools/yolov4.engine", std::ios::binary);
    
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    // prepare input data ---------------------------
    // static float data[3 * INPUT_H * INPUT_W];
    static float prob[2];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    // static float prob[OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    std::cout << context->getEngine().getNbOptimizationProfiles() << " profiles\n";
    context->setOptimizationProfile(0);
    assert(context != nullptr);
    delete[] trtModelStream;
    std::vector<BindInfo> bindinfos = getBindInfo(engine);
    int fcount = 0;
    std::vector<std::string> file_names;
    file_names.push_back("/media/wzq/Seagate/ubuntu1604/zhihuiyan/deepModelTraining/TAO/workspace/dataprocess/images/110.jpg");
   

    for (auto f: file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat img = cv::imread(f);
        int orig_w = img.cols;
        int orig_h = img.rows;
        float ratio = std::min(INPUT_W / float(orig_w), INPUT_H / float(orig_h));
        int new_w = int(round(orig_w * ratio));
        int new_h = int(round(orig_h * ratio));
        cv::Mat new_img;
        cv::resize(img, new_img, cv::Size(new_w, new_h));
        cv::Mat new_imgf;
        new_img.convertTo(new_imgf, CV_32FC3);
        cv::Mat model_input = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
//        model_input.setTo(255);
        cv::Mat A = model_input(cv::Range(0,new_h),cv::Range(0,new_w));
        new_imgf.copyTo(A);
        ratio = float(orig_w) / new_w;





        cv::Mat rgbimg;
        if (img.empty()) continue;
        cv::cvtColor(img, rgbimg, cv::COLOR_BGR2RGB);
        cv::Mat imgf;
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
        resized.convertTo(imgf,CV_32FC3, 1.0);
        cv::Mat subed = model_input - cv::Scalar(103.939, 116.779, 123.68);
        std::vector<cv::Mat>channles(3);
        cv::split(subed,channles);
        std::vector<float>data;
        float* ptr1 = (float*)(channles[0].data);
        float* ptr2 = (float*)(channles[1].data);
        float* ptr3 = (float*)(channles[2].data);
        data.insert(data.end(),ptr1,ptr1 + INPUT_H*INPUT_W);
        data.insert(data.end(),ptr2,ptr2 + INPUT_H*INPUT_W);
        data.insert(data.end(),ptr3,ptr3 + INPUT_H*INPUT_W);

        // Run inference
        auto start = std::chrono::system_clock::now();
        int output0;
        std::vector<std::vector<float>>other_outputs;
        doInference(*context,(float*)data.data(), 1, bindinfos,output0, other_outputs);
        std::vector<Object> objs = NvDsInferParseCustomBatchedNMSTLT(output0,other_outputs,0.6, INPUT_H, INPUT_W, img.rows, img.cols,
                                                                     ratio);
        std::cout << "Detected " << objs.size() << " odfs\n";
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        for(int i = 0;i < objs.size();i ++){
            cv::Rect r = cv::Rect(int(objs[i].left), int(objs[i].top), int(objs[i].width), int(objs[i].height));
            cv::rectangle(img, r, cv::Scalar(255, 0,0), 2, 8);
        }
        cv::imshow("win", img);
        cv::waitKey(0);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
