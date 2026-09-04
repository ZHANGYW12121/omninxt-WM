#!/usr/bin/env python3
"""TensorRT 8/10 runner and RTMPose adapter without PyCUDA."""

import ctypes

import numpy as np
import tensorrt as trt

from rtmlib import RTMPose, YOLOX
from rtmlib.tools.object_detection.post_processings import multiclass_nms


class CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

    @staticmethod
    def check(code, operation):
        if code != 0:
            raise RuntimeError("{} failed with CUDA error {}".format(operation, code))

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return pointer

    def free(self, pointer):
        if pointer and pointer.value:
            self.check(self.lib.cudaFree(pointer), "cudaFree")

    def copy_to_device(self, pointer, array):
        self.check(self.lib.cudaMemcpy(pointer, ctypes.c_void_p(array.ctypes.data),
                                       array.nbytes, self.HOST_TO_DEVICE), "cudaMemcpy H2D")

    def copy_to_host(self, array, pointer):
        self.check(self.lib.cudaMemcpy(ctypes.c_void_p(array.ctypes.data), pointer,
                                       array.nbytes, self.DEVICE_TO_HOST), "cudaMemcpy D2H")

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class TensorRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as stream:
            serialized = stream.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize " + engine_path)
        self.context = self.engine.create_execution_context()
        self.cuda = CudaRuntime()
        self.host = []
        self.device = []
        self.bindings = []
        self.tensor_names = []
        self.input_indices = []
        self.output_indices = []
        self.tensor_api = hasattr(self.engine, "num_io_tensors")
        count = (self.engine.num_io_tensors if self.tensor_api
                 else self.engine.num_bindings)
        for index in range(count):
            name = (self.engine.get_tensor_name(index) if self.tensor_api
                    else self.engine.get_binding_name(index))
            shape = tuple(int(value) for value in (
                self.engine.get_tensor_shape(name) if self.tensor_api
                else self.engine.get_binding_shape(index)))
            if any(value <= 0 for value in shape):
                raise ValueError("Only static TensorRT bindings are supported: {}".format(shape))
            trt_dtype = (self.engine.get_tensor_dtype(name) if self.tensor_api
                         else self.engine.get_binding_dtype(index))
            dtype_map = {
                trt.float32: np.float32,
                trt.float16: np.float16,
                trt.int8: np.int8,
                trt.int32: np.int32,
                trt.bool: np.bool_,
            }
            if trt_dtype not in dtype_map:
                raise TypeError("Unsupported TensorRT dtype: {}".format(trt_dtype))
            dtype = np.dtype(dtype_map[trt_dtype])
            array = np.empty(shape, dtype=dtype)
            pointer = self.cuda.malloc(array.nbytes)
            self.host.append(array)
            self.device.append(pointer)
            self.bindings.append(int(pointer.value))
            self.tensor_names.append(name)
            is_input = (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if self.tensor_api else self.engine.binding_is_input(index))
            (self.input_indices if is_input else self.output_indices).append(index)
            if self.tensor_api:
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError("set_tensor_address failed for " + name)
        if len(self.input_indices) != 1:
            raise ValueError("Expected one TensorRT input")

    def __del__(self):
        try:
            for pointer in getattr(self, "device", []):
                self.cuda.free(pointer)
        except Exception:
            pass

    def __call__(self, input_array):
        index = self.input_indices[0]
        source = np.ascontiguousarray(input_array, dtype=self.host[index].dtype)
        if source.shape != self.host[index].shape:
            raise ValueError("Expected {}, got {}".format(self.host[index].shape, source.shape))
        np.copyto(self.host[index], source)
        self.cuda.copy_to_device(self.device[index], self.host[index])
        if self.tensor_api:
            if not self.context.execute_async_v3(0):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self.cuda.synchronize()
        elif not self.context.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execute_v2 failed")
        outputs = []
        for output_index in self.output_indices:
            self.cuda.copy_to_host(self.host[output_index], self.device[output_index])
            outputs.append(self.host[output_index].copy())
        return outputs


class TensorRTRTMPose(RTMPose):
    def __init__(self, engine_path, model_input_size=(192, 256), to_openpose=False):
        self.runner = TensorRTRunner(engine_path)
        self.model_input_size = model_input_size
        self.mean = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
        self.std = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
        self.to_openpose = to_openpose

    def inference(self, image):
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.runner(tensor)


class TensorRTYOLOX(YOLOX):
    """YOLOX preprocessing/postprocessing with a raw-output TensorRT engine."""

    def __init__(self, engine_path, model_input_size=(416, 416),
                 det_mode="multiclass", nms_thr=0.45, score_thr=0.30):
        # Do not call BaseTool: the ONNX/ONNX Runtime session is deliberately
        # absent from the real-time path.
        self.runner = TensorRTRunner(engine_path)
        self.model_input_size = model_input_size
        self.det_mode = det_mode
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.onnx_model = None
        self.backend = "tensorrt"
        self.device = "cuda"

    def inference(self, image):
        tensor = np.ascontiguousarray(
            image.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.runner(tensor)

    def __call__(self, image):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)
        if len(outputs) == 1:
            return self.postprocess(outputs[0], ratio)
        if len(outputs) != 2:
            raise ValueError(
                "Expected raw YOLOX output or decoded boxes/scores, got {}"
                .format(len(outputs)))
        boxes = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(outputs[1], dtype=np.float32).reshape(
            len(boxes), -1)
        boxes /= float(ratio)
        # This runtime is exclusively a person front-end.  Applying NMS only
        # to COCO/HumanArt class 0 is both faster and avoids retaining a shelf
        # under a competing furniture class before the caller filters it.
        scores = scores[:, :1]
        detections, _ = multiclass_nms(
            boxes, scores, nms_thr=self.nms_thr,
            score_thr=self.score_thr)
        if detections is None:
            final_boxes = np.empty((0, 4), dtype=np.float32)
            final_classes = np.empty((0,), dtype=np.int32)
        else:
            final_boxes = detections[:, :4].astype(np.float32, copy=False)
            final_classes = detections[:, 5].astype(np.int32, copy=False)
        if self.det_mode == "multiclass":
            return final_boxes, final_classes
        if self.det_mode == "human":
            return final_boxes[final_classes == 0]
        raise NotImplementedError(
            "det_mode must be 'human' or 'multiclass': {}"
            .format(self.det_mode))
