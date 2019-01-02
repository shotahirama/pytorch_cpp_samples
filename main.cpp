#include <torch/script.h>

#include <iostream>
#include <memory>

#include <typeinfo>

#include <opencv2/opencv.hpp>

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

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  std::cout << argc << std::endl;
  torch::DeviceType device_type = at::kCPU;
  if (argc > 3) {
    device_type = at::kCUDA;
    std::cout << "use cuda" << std::endl;
  }

  torch::Device device(device_type);

  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  module->to(device);

  assert(module != nullptr);
  std::cout << "ok\n";

  cv::Mat img = cv::imread(argv[2]);
  cv::resize(img, img, cv::Size(224, 224));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  auto imgdata = reorder_to_chw(img);
  img = cv::Mat(imgdata, CV_32FC3);

  torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, torch::dtype(torch::kFloat32)).view(at::IntList{3, 1, 1}).to(device);
  torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}, torch::dtype(torch::kFloat32)).view(at::IntList{3, 1, 1}).to(device);

  std::vector<torch::jit::IValue> inputs;
  torch::Tensor input = torch::from_blob(img.ptr<float>(), {1, 3, 224, 224}).to(device);
  // torch::Tensor input = torch::from_blob(img.ptr<float>(), {3, 224, 224});
  // input = input.unsqueeze(0);
  input = input.sub(mean).div(std);
  inputs.push_back(input);

  at::Tensor output = module->forward(inputs).toTensor();
  std::cout << output.argmax().item<int>() << std::endl;
  std::cout << output.flatten()[0].item<float>() << std::endl;
}
