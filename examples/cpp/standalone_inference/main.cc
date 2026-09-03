// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_cpu/kernels/kernel_usage.h"
#include "onnx_light_cpu/kernels/register_kernels.h"

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/stream.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ol = ONNX_LIGHT_NAMESPACE;
namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

ol::ModelProto MakeAbsModel() {
  ol::ModelProto model;
  model.set_ir_version(10);
  ol::OperatorSetIdProto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);

  ol::GraphProto *graph = model.add_graph();
  graph->set_name("standalone_abs");
  graph->add_input()->set_name("x");
  graph->add_output()->set_name("y");

  ol::NodeProto *node = graph->add_node();
  node->set_op_type("Abs");
  node->add_input("x");
  node->add_output("y");
  return model;
}

ol::ModelProto LoadModel(const std::string &path) {
  ol::ModelProto model;
  ol::utils::FileStream stream(path);
  ol::ParseOptions options;
  ol::ParseModelProtoFromStream(model, stream, options);
  return model;
}

std::string FirstInputName(const ol::ModelProto &model) {
  if (!model.has_graph() || model.ref_graph().ref_input().empty()) {
    throw std::invalid_argument("the model must declare at least one graph input");
  }
  return model.ref_graph().ref_input()[0].ref_name();
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2) {
      std::cerr << "Usage: " << argv[0] << " [model.onnx]\n";
      return 2;
    }

    ol::ModelProto model = argc == 2 ? LoadModel(argv[1]) : MakeAbsModel();
    const std::string input_name = FirstInputName(model);

    onnx_light_cpu::RegisterAllKernelsGlobal();
    onnx_light_cpu::ClearUsedKernelNames();

    rt::Tensors inputs;
    inputs.push_back(rt::Tensor::FromFloat(input_name, {4}, {-1.0f, 2.0f, -3.5f, 4.0f}));
    const rt::Tensors outputs = rt::RunModel(model, std::move(inputs));
    if (outputs.size() != 1 || outputs[0].element_count() != 4) {
      throw std::runtime_error("the example expects one four-element tensor output");
    }

    const float *values = outputs[0].AsFloat();
    std::cout << outputs[0].name << " = [";
    for (std::size_t index = 0; index < 4; ++index) {
      std::cout << (index == 0 ? "" : ", ") << values[index];
    }
    std::cout << "]\n";

    const std::vector<std::string> used_kernels = onnx_light_cpu::UsedKernelNames();
    constexpr const char *expected_kernel = "onnx_light_cpu::Abs";
    if (std::find(used_kernels.begin(), used_kernels.end(), expected_kernel) ==
        used_kernels.end()) {
      throw std::runtime_error("the model did not dispatch to the onnx-light-cpu Abs kernel");
    }

    const float expected[] = {1.0f, 2.0f, 3.5f, 4.0f};
    for (std::size_t index = 0; index < 4; ++index) {
      if (std::fabs(values[index] - expected[index]) > 1e-6f) {
        throw std::runtime_error("unexpected inference output");
      }
    }
    std::cout << "Kernel used: " << expected_kernel << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Inference failed: " << error.what() << "\n";
    return 1;
  }
}
