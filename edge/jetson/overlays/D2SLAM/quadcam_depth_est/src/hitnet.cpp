#include "hitnet.hpp"
#include <iostream>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/logger.h"
#include "tensorrt_utils/common.h"

namespace TensorRTHitnet{
int32_t HitnetTrt::init(const std::string& onnx_model_path, const std::string& trt_engine_path, int32_t stream_number){
  if(stream_number <= 0){
    stream_number = 1;
  }
  stream_number_ = stream_number;

  //Check TRT engine file can be loaded, if not create engine from onnx model and save to trt_engine_path
  if (access(trt_engine_path.c_str(), F_OK) == -1){
    spdlog::info("TRT engine file not found, create engine from onnx model");
    if (access(onnx_model_path.c_str(), F_OK) == -1){
      spdlog::error("onnx model file not found");
      return -2;
    }
    if (buildEngine(onnx_model_path, trt_engine_path) != 0){
      spdlog::error("buildEngine failed");
      return -3;
    }
  } else {
    spdlog::info("TRT engine file found, load engine from file");
    if (deserializeEngine(trt_engine_path) != 0){
      spdlog::error("deserializeEngine failed");
      return -4;
    }
  }

  //Create executors
  for(int32_t i = 0; i < stream_number_; i++){
    auto executor = std::make_unique<HitnetExcutor>();
    int32_t ret = executor->init(this->nv_engine_ptr_,std::string(kInputTensorName),std::string(kOutputTensorName));
    if(ret != 0){
      spdlog::error("Hitnet multistream Excutor init failed");
      return -5;
    }
    this->executors_.push_back(std::move(executor));
  }
  printf("Success to init HitnetTrt\n");
  return 0;
}

//TODO: inference stucked
int32_t HitnetTrt::doInference(const cv::Mat input[4]){
  for(int32_t i = 0; i < this->stream_number_; i++){
    if(input[i].empty()){
      return 0;
    }
    int32_t ret = this->executors_[i]->setInputImages(input[i]);
    if(ret != 0){
      std::cout << "setInputImages failed" << std::endl;
      return -2;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i]->doInference();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i]->copyBack();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i]->synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -4;
    }
  }

  #ifdef DEBUG
  printf ("inferenced\n");
  #endif

  return 0;
}

int32_t HitnetTrt::getOutput(cv::Mat output[4]){
  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i]->getOutput(output[i]);
    if(ret != 0){
      std::cout << "getOutput failed" << std::endl;
      return -5;
    }
  }
  return 0;
}

HitnetTrt::~HitnetTrt(){
  executors_.clear();
  nv_engine_ptr_.reset();
  nv_runtime_ptr_.reset();
}


int32_t HitnetTrt::deserializeEngine(const std::string& trt_engine_path){
  std::ifstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    spdlog::info("load engine from file: {}", trt_engine_path);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();
    nv_runtime_ptr_ = std::shared_ptr<nvinfer1::IRuntime>(
        createInferRuntime(tensorrt_log::gLogger),
        [](nvinfer1::IRuntime* ptr) { if (ptr) ptr->destroy(); });
    if (!nv_runtime_ptr_) {
      spdlog::error("createInferRuntime failed");
      return -3;
    }
    this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        nv_runtime_ptr_->deserializeCudaEngine(engine_data.data(), size, nullptr),
        [](nvinfer1::ICudaEngine* ptr) { if (ptr) ptr->destroy(); });
    if(this->nv_engine_ptr_ == nullptr){
      spdlog::error("deserializeCudaEngine failed");
      return -3;
    }
    spdlog::info("Success to load engine from file");
    return 0;
  } else {
    return -2;
  }
}

int32_t HitnetTrt::buildEngine(const std::string& onnx_model_path, const std::string& trt_engine_path){
  auto builder = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(tensorrt_log::gLogger.getTRTLogger()));
  if (builder == nullptr){
    spdlog::error("createInferBuilder failed");
    return -1;
  }
  auto network = tensorrt_common::TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
    1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (network == nullptr){
    spdlog::error("createNetworkV2 failed");
    return -2;
  }
  auto config = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (config == nullptr){
    spdlog::error("createBuilderConfig failed");
    return -3;
  }
  auto parser = tensorrt_common::TensorRTUniquePtr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, tensorrt_log::gLogger.getTRTLogger()));
  if (parser == nullptr){
    spdlog::error("createParser failed");
    return -4;
  }
  auto parsed = parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(tensorrt_log::gLogger.getReportableSeverity()));
  if (!parsed){
    spdlog::error("parseFromFile failed");
    return -6;
  }
  if (network->getNbInputs() != 1) {
    spdlog::error("Expected one HITNet input, got {}", network->getNbInputs());
    return -6;
  }
  const nvinfer1::Dims4 expected_input{1, 2, 240, 320};
  auto* input_tensor = network->getInput(0);
  auto input_dims = input_tensor->getDimensions();
  bool dynamic_input = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    dynamic_input = dynamic_input || input_dims.d[i] == -1;
  }
  if (dynamic_input) {
    auto profile = builder->createOptimizationProfile();
    if (profile == nullptr ||
        !profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, expected_input) ||
        !profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, expected_input) ||
        !profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, expected_input) ||
        config->addOptimizationProfile(profile) < 0) {
      spdlog::error("Failed to configure the dynamic HITNet input profile");
      return -5;
    }
  } else {
    if (input_dims.nbDims != expected_input.nbDims) {
      spdlog::error("Unexpected HITNet input rank {}", input_dims.nbDims);
      return -5;
    }
    for (int i = 0; i < input_dims.nbDims; ++i) {
      if (input_dims.d[i] != expected_input.d[i]) {
        spdlog::error("Unexpected static HITNet input dimension at {}: {}", i,
                      input_dims.d[i]);
        return -5;
      }
    }
  }
  config ->setMaxWorkspaceSize(1024_MiB);
  config ->setFlag(nvinfer1::BuilderFlag::kFP16);
  config ->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

  auto profile_stream = tensorrt_common::makeCudaStream();
  if (profile_stream == nullptr){
    spdlog::error("makeCudaStream failed");
    return -7;
  }
  config->setProfileStream(*profile_stream);
  tensorrt_common::TensorRTUniquePtr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr){
    spdlog::error("buildSerializedNetwork failed");
    return -8;
  }
  nv_runtime_ptr_ = std::shared_ptr<nvinfer1::IRuntime>(
      createInferRuntime(tensorrt_log::gLogger.getTRTLogger()),
      [](nvinfer1::IRuntime* ptr) { if (ptr) ptr->destroy(); });
  if (nv_runtime_ptr_ == nullptr){
    spdlog::error("createInferRuntime failed");
    return -9;
  }
  this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      nv_runtime_ptr_->deserializeCudaEngine(plan->data(), plan->size(), nullptr),
      [](nvinfer1::ICudaEngine* ptr) { if (ptr) ptr->destroy(); });
  if (this->nv_engine_ptr_ == nullptr){
    spdlog::error("deserializeCudaEngine failed");
    return -10;
  }

  //Save engine to file
  std::ofstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    engine_file.write(static_cast<const char*>(plan->data()), plan->size());
    engine_file.close();
    spdlog::info("Success to save engine to file: {}", trt_engine_path);
  } else {
    spdlog::error("Failed to save engine to file: {}", trt_engine_path);
    return -11;
  }
  return 0;
}



HitnetExcutor::~HitnetExcutor(){
  buffer_manager_ptr_.reset();
  if (nv_context_ptr_ != nullptr) {
    nv_context_ptr_->destroy();
    nv_context_ptr_ = nullptr;
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  engine_ptr_.reset();
}

int32_t HitnetExcutor::init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr,
  std::string input_tensor_name,
  std::string output_tensor_name){
  this->engine_ptr_ = engine_ptr;
  this->nv_context_ptr_ = this->engine_ptr_->createExecutionContext();
  if(this->nv_context_ptr_ == nullptr){
    std::cout << "createExecutionContext failed" << std::endl;
    return -1;
  }
  if(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking) != cudaSuccess){
    std::cout << "cudaStreamCreateWithFlags failed" << std::endl;
    return -2;
  }
  this->buffer_manager_ptr_ = std::make_unique<tensorrt_buffer::BufferManager>(this->engine_ptr_,
    0,this->nv_context_ptr_);
  if(this->buffer_manager_ptr_ == nullptr){
    std::cout << "make_unique BufferManager failed" << std::endl;
    return -3;
  }

  this->input_tensor_name_ = input_tensor_name;
  this->output_tensor_name_ = output_tensor_name;
  this->input_index_ = this->engine_ptr_->getBindingIndex(input_tensor_name.c_str());
  if(this->input_index_ < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -4;
  }
  auto input_dim = this->engine_ptr_->getBindingDimensions(this->input_index_);
  if (this->engine_ptr_->getBindingDataType(this->input_index_) !=
          nvinfer1::DataType::kFLOAT ||
      input_dim.nbDims != 4 || input_dim.d[0] != 1 ||
      input_dim.d[1] != 2 || input_dim.d[2] != 240 ||
      input_dim.d[3] != 320) {
    std::cout << "Unexpected HITNet input; expected float32 [1,2,240,320]"
              << std::endl;
    return -5;
  }
  this->input_size_ = this->buffer_manager_ptr_->size(this->input_tensor_name_);
  this->output_index_ = this->engine_ptr_->getBindingIndex(output_tensor_name.c_str());
  if(this->output_index_  < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -5;
  }
  auto output_dim = this->engine_ptr_->getBindingDimensions(this->output_index_ );
  if (this->engine_ptr_->getBindingDataType(this->output_index_) !=
          nvinfer1::DataType::kFLOAT ||
      output_dim.nbDims != 4 || output_dim.d[0] != 1 ||
      output_dim.d[1] != 240 || output_dim.d[2] != 320 ||
      output_dim.d[3] != 1) {
    std::cout << "Unexpected HITNet output; expected float32 [1,240,320,1]"
              << std::endl;
    return -6;
  }
  this->output_size_ = this->buffer_manager_ptr_->size(this->output_tensor_name_);
  this->output_height_ = output_dim.d[1];
  this->output_width_ = output_dim.d[2];

  // Warm TensorRT once before capture: some tactics allocate resources on the
  // first enqueue and are not capture-safe. This runs during node
  // construction, before OpenCV's rectification thread starts using CUDA.
  buffer_manager_ptr_->copyInputToDeviceAsync(stream_);
  if (!nv_context_ptr_->enqueueV2(
          buffer_manager_ptr_->getDeviceBindings().data(), stream_, nullptr)) {
    std::cout << "HITNet warmup enqueueV2 failed" << std::endl;
    return -7;
  }
  buffer_manager_ptr_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cout << "HITNet warmup synchronize failed" << std::endl;
    return -7;
  }

  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess) {
    std::cout << "cudaStreamBeginCapture failed" << std::endl;
    return -8;
  }
  buffer_manager_ptr_->copyInputToDeviceAsync(stream_);
  const bool capture_status = nv_context_ptr_->enqueueV2(
      buffer_manager_ptr_->getDeviceBindings().data(), stream_, nullptr);
  buffer_manager_ptr_->copyOutputToHostAsync(stream_);
  if (!capture_status || cudaStreamEndCapture(stream_, &graph_) != cudaSuccess ||
      graph_ == nullptr) {
    std::cout << "HITNet CUDA graph capture failed" << std::endl;
    return -8;
  }
  if (cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0) !=
          cudaSuccess ||
      graph_exec_ == nullptr) {
    std::cout << "HITNet CUDA graph instantiate failed" << std::endl;
    return -8;
  }
  return 0;
}

int32_t HitnetExcutor::setInputImages(const cv::Mat& input){
  if (!input.isContinuous() || input.depth() != CV_32F ||
      input.total() * input.elemSize() != static_cast<size_t>(input_size_)) {
    std::cout << "Unexpected HITNet input buffer layout or size" << std::endl;
    return -1;
  }
  memcpy(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name_), input.data, this->input_size_);
  return 0;
}

int32_t HitnetExcutor::doInference(){
  if (graph_exec_ == nullptr ||
      cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess) {
    std::cout << "HITNet CUDA graph launch failed" << std::endl;
    return -1;
  }
  return 0;
}

int32_t HitnetExcutor::copyBack(){
  // The graph contains H2D, TensorRT enqueue, and D2H operations.
  return 0;
}

int32_t HitnetExcutor::synchronize(){
  return cudaStreamSynchronize(this->stream_);
}

int32_t HitnetExcutor::getOutput(cv::Mat& output){
  if (output.empty() || output.rows != output_height_ ||
      output.cols != output_width_ || output.type() != CV_32F) {
    output = cv::Mat(output_height_, output_width_, CV_32F);
  }
  memcpy(output.data, this->buffer_manager_ptr_->getHostBuffer(this->output_tensor_name_), this->output_size_);
  return 0;
}
}
