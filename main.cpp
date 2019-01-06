#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<float> reorder_to_chw(cv::Mat const& img) {
  assert(img.channels() == 3);

  cv::Mat mat = img.clone();

  std::vector<float> data(mat.channels() * mat.rows * mat.cols);

  for (int y = 0; y < mat.rows; ++y) {
    for (int x = 0; x < mat.cols; ++x) {
      for (int c = 0; c < mat.channels(); ++c) {
        data[c * (mat.rows * mat.cols) + y * mat.cols + x] =

            static_cast<unsigned int>(mat.data[y * mat.step + x * mat.elemSize() + c]) / 255.0;
      }
    }
  }

  return data;
}

std::vector<std::string> split_labelname(std::string filepath) {
  std::ifstream ifs(filepath);
  std::string line;
  std::vector<std::string> labelnames;
  while (std::getline(ifs, line)) {
    auto p1 = line.find("'") + 1;
    auto p2 = line.rfind("'");
    labelnames.emplace_back(line.substr(p1, p2 - p1));
  }
  return labelnames;
}

int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <imagenet-labels-txt> <image>\n";
    return -1;
  }

  torch::DeviceType device_type = at::kCPU;
  if (argc > 4) {
    device_type = at::kCUDA;
    std::cout << "use cuda" << std::endl;
  }

  torch::Device device(device_type);

  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  module->to(device);

  assert(module != nullptr);

  auto labelnames = split_labelname(argv[2]);
  cv::Mat img = cv::imread(argv[3]);
  cv::resize(img, img, cv::Size(224, 224));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  auto imgdata = reorder_to_chw(img);
  img = cv::Mat(imgdata, CV_32FC3);

  torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, torch::dtype(torch::kFloat32)).view(at::IntList{3, 1, 1}).to(device);
  torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}, torch::dtype(torch::kFloat32)).view(at::IntList{3, 1, 1}).to(device);

  std::vector<torch::jit::IValue> inputs;
  torch::Tensor input = torch::from_blob(img.ptr<float>(), {1, 3, 224, 224}).to(device);
  // torch::Tensor input = torch::from_blob(img.ptr<float>(), {3, 224, 224}).unsqueeze(0);
  input = input.sub(mean).div(std);
  inputs.push_back(input);

  at::Tensor output = module->forward(inputs).toTensor();
  int label = output.argmax().item<int>();
  std::cout << labelnames[label] << std::endl;
}
