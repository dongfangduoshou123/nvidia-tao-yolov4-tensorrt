#! python3
from pydoc import importfile
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
# import cv2
serialized_engine_path = "/opt/deepstream-dev/saved.engine"
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
# const char* INPUT_BLOB_NAME = "input_1";
# const char* OUTPUT_BLOB_NAME1 = "predictions/Softmax";
input_name = "input_1"
output_name = "predictions/Softmax"

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        batch_size = 1
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        # engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

import PIL
from PIL import Image
import numpy as np

def _load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, 3)
        ).astype(np.uint8)


def _load_img(image_path):
    image = Image.open(image_path)
    model_input_width = 224
    model_input_height = 224
    # Note: Bilinear interpolation used by Pillow is a little bit
    # different than the one used by Tensorflow, so if network receives
    # an image that is not 300x300, the network output may differ
    # from the one output by Tensorflow
    image_resized = image.resize(
        size=(model_input_width, model_input_height),
        resample=Image.BILINEAR
    )
    img_np = _load_image_into_numpy_array(image_resized)
    # HWC -> CHW
    img_np = img_np.transpose((2, 0, 1))
    # Normalize to [-1.0, 1.0] interval (expected by model)
    img_np = 1.0 * img_np
    img_np = img_np.ravel()
    return img_np

def preprare_img(img_path):
    """
        cv::Mat img = cv::imread(f);
        cv::Mat rgbimg;
        if (img.empty()) continue;
        cv::cvtColor(img, rgbimg, cv::COLOR_BGR2RGB);
        cv::Mat imgf;
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
        resized.convertTo(imgf,CV_32FC3, 1.0);
    """
    pass
    # mat = cv2.imread(img_path)
    # cv2.cvtColor(mat, rgbimg, cv2.COLOR_BGR2RGB)
    # img_np = img_np.transpose((2, 0, 1))
    # img_np = 1.0 * img_np
    # img_np = img_np.ravel()

with open(serialized_engine_path, "rb") as f:
    serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    input_idx = engine[input_name]
    output_idx = engine[output_name]
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    img = _load_img("/opt/deepstream-dev/2.jpg")
    np.copyto(inputs[0].host, img.ravel())
    ret = do_inference(context, bindings, inputs, outputs, stream)
    print(ret)
