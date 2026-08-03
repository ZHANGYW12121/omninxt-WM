#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <iostream>
#include <memory>
#include <string>

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

template <typename T>
struct Destroy {
  void operator()(T* value) const {
    if (value != nullptr) {
      value->destroy();
    }
  }
};

static void printDims(const nvinfer1::Dims& dims) {
  std::cout << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i != 0) std::cout << ',';
    std::cout << dims.d[i];
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: inspect_onnx_trt MODEL.onnx\n";
    return 2;
  }
  Logger logger;
  std::unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder(
      nvinfer1::createInferBuilder(logger));
  if (!builder) return 3;
  const auto flags =
      1U << static_cast<unsigned>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition,
                  Destroy<nvinfer1::INetworkDefinition>>
      network(builder->createNetworkV2(flags));
  std::unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser(
      nvonnxparser::createParser(*network, logger));
  if (!network || !parser || !parser->parseFromFile(argv[1], 2)) {
    std::cerr << "ONNX parse failed\n";
    return 4;
  }
  std::cout << "inputs=" << network->getNbInputs()
            << " outputs=" << network->getNbOutputs() << '\n';
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto* tensor = network->getInput(i);
    std::cout << "input[" << i << "] name=" << tensor->getName() << " dims=";
    printDims(tensor->getDimensions());
    std::cout << " dtype=" << static_cast<int>(tensor->getType()) << '\n';
  }
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    auto* tensor = network->getOutput(i);
    std::cout << "output[" << i << "] name=" << tensor->getName() << " dims=";
    printDims(tensor->getDimensions());
    std::cout << " dtype=" << static_cast<int>(tensor->getType()) << '\n';
  }
  return 0;
}
