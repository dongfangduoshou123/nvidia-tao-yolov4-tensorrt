#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <map>


static std::map<int, std::string> index_label = {
{0, "hard"},
{1, "middle"},
{2, "simple"}
};

extern "C"
bool NvDsInferParseCustomClassify(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString){
    assert(outputLayersInfo.size() ==1);
    float* out_softmax = (float *) outputLayersInfo[0].buffer;
    // for(int i = 0;i < outputLayersInfo[0].inferDims.numDims; i ++){
    //     printf("outputDim index i:%d size:%d\n",i, outputLayersInfo[0].inferDims.d[i]);
    // }
    // printf("output values: %f, %f, %f\n", out_softmax[0], out_softmax[1], out_softmax[1]);
    int index = (out_softmax[0] > out_softmax[1] ? 0: 1);
    index = (out_softmax[2] > out_softmax[index] ? 2: index);
    // printf("The max index id is %d\n", index);
    NvDsInferAttribute attr;
    attr.attributeConfidence = out_softmax[index];
    attr.attributeIndex = index;
    attr.attributeValue = index;
    attr.attributeLabel = strdup(index_label[index].c_str());
    attrList.push_back(attr);
    descString = index_label[index];
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomClassify)
