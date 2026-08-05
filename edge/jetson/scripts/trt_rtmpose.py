#!/usr/bin/env python3
"""Minimal TensorRT 8 runner and RTMPose adapter without PyCUDA."""

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
        self.binding_dtypes = []
        self.capacity_bytes = []
        self.input_indices = []
        self.output_indices = []
        dtype_map = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int8: np.int8,
            trt.int32: np.int32,
            trt.bool: np.bool_,
        }
        for index in range(self.engine.num_bindings):
            trt_dtype = self.engine.get_binding_dtype(index)
            if trt_dtype not in dtype_map:
                raise TypeError("Unsupported TensorRT dtype: {}".format(trt_dtype))
            self.binding_dtypes.append(np.dtype(dtype_map[trt_dtype]))
            (self.input_indices if self.engine.binding_is_input(index)
             else self.output_indices).append(index)
        if len(self.input_indices) != 1:
            raise ValueError("Expected one TensorRT input")
        input_index = self.input_indices[0]
        declared = tuple(int(value) for value in
                         self.engine.get_binding_shape(input_index))
        self.dynamic = any(value <= 0 for value in declared)
        if self.dynamic:
            minimum, optimum, maximum = self.engine.get_profile_shape(
                0, input_index)
            self.min_input_shape = tuple(int(value) for value in minimum)
            self.opt_input_shape = tuple(int(value) for value in optimum)
            self.max_input_shape = tuple(int(value) for value in maximum)
            if not self.context.set_binding_shape(
                    input_index, self.max_input_shape):
                raise RuntimeError("Failed to select maximum TensorRT shape")
        else:
            if any(value <= 0 for value in declared):
                raise ValueError("Invalid TensorRT input shape: {}".format(
                    declared))
            self.min_input_shape = declared
            self.opt_input_shape = declared
            self.max_input_shape = declared
        self.max_batch_size = int(self.max_input_shape[0])
        self.min_batch_size = int(self.min_input_shape[0])
        # Allocate once for the largest profile shape.  Calls with a smaller
        # batch copy only their actual byte count into the same buffers.
        for index in range(self.engine.num_bindings):
            shape = tuple(int(value) for value in
                          self.context.get_binding_shape(index))
            if any(value <= 0 for value in shape):
                raise ValueError(
                    "Unable to resolve TensorRT binding {} shape {}".format(
                        index, shape))
            array = np.empty(shape, dtype=self.binding_dtypes[index])
            pointer = self.cuda.malloc(array.nbytes)
            self.host.append(array)
            self.device.append(pointer)
            self.bindings.append(int(pointer.value))
            self.capacity_bytes.append(array.nbytes)
        if self.dynamic and not self.context.set_binding_shape(
                input_index, self.opt_input_shape):
            raise RuntimeError("Failed to select optimum TensorRT shape")

    def __del__(self):
        try:
            for pointer in getattr(self, "device", []):
                self.cuda.free(pointer)
        except Exception:
            pass

    def __call__(self, input_array):
        index = self.input_indices[0]
        source = np.ascontiguousarray(
            input_array, dtype=self.binding_dtypes[index])
        if self.dynamic:
            if len(source.shape) != len(self.max_input_shape) or any(
                    actual < low or actual > high
                    for actual, low, high in zip(
                        source.shape, self.min_input_shape,
                        self.max_input_shape)):
                raise ValueError(
                    "Expected a shape between {} and {}, got {}".format(
                        self.min_input_shape, self.max_input_shape,
                        source.shape))
            if not self.context.set_binding_shape(index, source.shape):
                raise RuntimeError(
                    "TensorRT rejected input shape {}".format(source.shape))
            if not self.context.all_binding_shapes_specified:
                raise RuntimeError("Not all TensorRT binding shapes are set")
        elif source.shape != self.max_input_shape:
            raise ValueError("Expected {}, got {}".format(
                self.max_input_shape, source.shape))
        if source.nbytes > self.capacity_bytes[index]:
            raise ValueError("Input exceeds allocated TensorRT buffer")
        self.cuda.copy_to_device(self.device[index], source)
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execute_v2 failed")
        outputs = []
        for output_index in self.output_indices:
            shape = tuple(int(value) for value in
                          self.context.get_binding_shape(output_index))
            if any(value <= 0 for value in shape):
                raise RuntimeError(
                    "Unresolved TensorRT output shape {}".format(shape))
            output = np.empty(shape, dtype=self.binding_dtypes[output_index])
            if output.nbytes > self.capacity_bytes[output_index]:
                raise ValueError("Output exceeds allocated TensorRT buffer")
            self.cuda.copy_to_host(output, self.device[output_index])
            outputs.append(output)
        return outputs


class TensorRTRTMPose(RTMPose):
    def __init__(self, engine_path, model_input_size=(192, 256), to_openpose=False):
        self.runner = TensorRTRunner(engine_path)
        self.model_input_size = model_input_size
        self.mean = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
        self.std = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
        self.to_openpose = to_openpose
        self.last_batch_sizes = ()
        self.last_person_count = 0

    def __call__(self, image, bboxes=[]):
        """Run all person crops in the fewest TensorRT batch calls possible."""
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]
        crops = []
        metadata = []
        for bbox in bboxes:
            crop, center, scale = self.preprocess(image, bbox)
            crops.append(np.ascontiguousarray(
                crop.transpose(2, 0, 1), dtype=np.float32))
            metadata.append((center, scale))
        keypoints = []
        scores = []
        batch_sizes = []
        capacity = max(1, self.runner.max_batch_size)
        for start in range(0, len(crops), capacity):
            stop = min(len(crops), start + capacity)
            tensor = np.stack(crops[start:stop], axis=0)
            outputs = self.runner(tensor)
            actual_batch = stop - start
            if any(output.shape[0] != actual_batch for output in outputs):
                raise RuntimeError("Unexpected RTMPose batch output shapes: {}"
                                   .format([value.shape for value in outputs]))
            for local_index in range(actual_batch):
                center, scale = metadata[start + local_index]
                sliced = [value[local_index:local_index + 1]
                          for value in outputs]
                points, confidence = self.postprocess(
                    sliced, center, scale)
                keypoints.append(points)
                scores.append(confidence)
            batch_sizes.append(actual_batch)
        self.last_batch_sizes = tuple(batch_sizes)
        self.last_person_count = len(crops)
        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        if self.to_openpose:
            from rtmlib.tools.pose_estimation.post_processings import \
                convert_coco_to_openpose
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)
        return keypoints, scores

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
        self.last_scores = np.empty((0,), dtype=np.float32)
        self.onnx_model = None
        self.backend = "tensorrt"
        self.device = "cuda"

    def inference(self, image):
        tensor = np.ascontiguousarray(
            image.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.runner(tensor)

    def postprocess(self, outputs, ratio=1.0):
        """Decode raw YOLOX output and retain the true detection scores.

        The bundled rtmlib implementation performs multiclass NMS with
        ``score_thr`` and then incorrectly filters the survivors again using
        ``score > nms_thr``.  With the normal 0.28/0.45 configuration that
        silently turns the effective person threshold into 0.45 and removes
        many small distant people.  NMS overlap and confidence are independent
        quantities, so no second confidence filter belongs here.
        """
        raw = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        raw = np.asarray(raw)
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[-1] <= 5:
            raise ValueError("Expected raw YOLOX output [1,N,5+C], got {}"
                             .format(raw.shape))

        grids = []
        expanded_strides = []
        for stride in (8, 16, 32):
            hsize = self.model_input_size[0] // stride
            wsize = self.model_input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        grids = np.concatenate(grids, axis=1)
        expanded_strides = np.concatenate(expanded_strides, axis=1)

        decoded = raw.copy()
        decoded[..., :2] = (decoded[..., :2] + grids) * expanded_strides
        decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * expanded_strides
        predictions = decoded[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * .5
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * .5
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * .5
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * .5
        boxes_xyxy /= float(ratio)

        detections, _ = multiclass_nms(
            boxes_xyxy, scores, nms_thr=self.nms_thr,
            score_thr=self.score_thr)
        if detections is None:
            self.last_scores = np.empty((0,), dtype=np.float32)
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.int32))
        final_boxes = detections[:, :4].astype(np.float32, copy=False)
        self.last_scores = detections[:, 4].astype(np.float32, copy=True)
        final_classes = detections[:, 5].astype(np.int32, copy=False)
        return final_boxes, final_classes
