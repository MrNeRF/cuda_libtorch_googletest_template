#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>

struct Libtorch_Simple_Net : torch::nn::Module {
    Libtorch_Simple_Net(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(inputDim1, outputDim1));
        fc2 = register_module("fc2", torch::nn::Linear(inputDim2, outputDim2));
        fc3 = register_module("fc3", torch::nn::Linear(inputDim3, outputDim3));
    }

    // Implement the Net's algorithm.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        torch::Tensor x1, x2, x3;

        x1 = torch::relu(fc1->forward(x));
        x2 = torch::relu(fc2->forward(x1));
        x3 = fc3->forward(x2);

        return std::make_tuple(x1, x2, x3);
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TEST(ForwardPassLossLibtorch, BasicTest) {
    std::vector<int> labels{0, 1, 0, 1, 0};
    std::vector<float> input{0.1f, 0.1f, 0.1f, 0.1f,
                             0.2f, 0.2f, 0.2f, 0.2f,
                             0.3f, 0.3f, 0.3f, 0.3f,
                             0.4f, 0.4f, 0.4f, 0.4f,
                             0.5f, 0.5f, 0.5f, 0.5f};

    float h_weights1[12] = {-.1f, .2f, .2f, .2f,
                            .5f, -.6f, .7f, .8f,
                            .9f, .10f, -.11f, .12f};
    float h_biases1[3] = {.1f, .2f, .3f};

    float h_weights2[9] = {-.1f, .2f, .3f,
                           .4f, .5f, -.6f,
                           .7f, .8f, .9f};
    float h_biases2[3] = {.4f, .2f, .3f};

    float h_weights3[6] = {-.1f, .2f, .3f,
                           .4f, .5f, -.6f};
    float h_biases3[2] = {.1f, .2f};

    // Prepare device memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto inputTensor = torch::from_blob(input.data(), {5, 4}, options).requires_grad_(true);
    auto weightTensor1 = torch::from_blob(h_weights1, {3, 4}, options).requires_grad_(true);
    auto biasTensor1 = torch::from_blob(h_biases1, {3}, options).requires_grad_(true);
    auto weightTensor2 = torch::from_blob(h_weights2, {3, 3}, options).requires_grad_(true);
    auto biasTensor2 = torch::from_blob(h_biases2, {3}, options).requires_grad_(true);
    auto weightTensor3 = torch::from_blob(h_weights3, {2, 3}, options).requires_grad_(true);
    auto biasTensor3 = torch::from_blob(h_biases3, {2}, options).requires_grad_(true);

    auto torchNet = std::make_shared<Libtorch_Simple_Net>(4, 3,
                                                          3, 3,
                                                          3, 2);

    torchNet->fc1->weight = weightTensor1;
    torchNet->fc1->bias = biasTensor1;
    torchNet->fc2->weight = weightTensor2;
    torchNet->fc2->bias = biasTensor2;
    torchNet->fc3->weight = weightTensor3;
    torchNet->fc3->bias = biasTensor3;

    auto [pred1, pred2, pred3] = torchNet->forward(inputTensor);

    // This step is necessary because torch is picky about the input type
    // and the labels are int64_t. If you insert std::vector<int>, the loss computation explodes.
    std::vector<int64_t> labels_torch_long(labels.begin(), labels.end());

    auto tensorLables = torch::from_blob(labels_torch_long.data(), {5}, torch::TensorOptions().dtype(torch::kLong));
    auto libtorch_loss = torch::nn::functional::cross_entropy(pred3, tensorLables);
    EXPECT_NEAR(0.70234048366546631f, libtorch_loss.item<float>(), 1e-4);

    torch::optim::SGD optimizer(torchNet->parameters(), /*lr=*/0.01);
    optimizer.zero_grad();
    libtorch_loss.backward();
    optimizer.step();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
