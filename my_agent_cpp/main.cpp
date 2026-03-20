#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "[INFO] Starting agent inference..." << std::endl;

    // === Load the model ===
    torch::jit::script::Module model;
    try {
        std::cout << "[INFO] Trying to load model from agent_script.pt" << std::endl;
        model = torch::jit::load("agent_script.pt");
        std::cout << "[INFO] Model loaded!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] c10::Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] std::exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception!" << std::endl;
        return -1;
    }


    model.eval();
    std::cout << "[INFO] Set model to eval mode." << std::endl;

    //Current CartPole v1 Env's Dimensions and hidden size
    const int obs_dim = 4;
    const int hidden_size = 128;
    const int num_layers = 1;

    std::cout << "[INFO] Preparing dummy inputs..." << std::endl;

    std::vector<float> obs_vec(obs_dim, 0.5f);
    torch::Tensor obs = torch::from_blob(obs_vec.data(), {1, obs_dim}).clone();
    torch::Tensor h_0 = torch::zeros({num_layers, 1, hidden_size});
    torch::Tensor c_0 = torch::zeros({num_layers, 1, hidden_size});
    torch::Tensor done = torch::zeros({1});

    std::cout << "[INFO] Inputs prepared." << std::endl;

    // === Forward pass ===
    try {
        std::cout << "[INFO] Running model.forward..." << std::endl;
        std::vector<torch::jit::IValue> inputs = {obs, h_0, c_0, done};
        auto output = model.forward(inputs).toTuple();

        std::cout << "[INFO] Extracting outputs..." << std::endl;
        torch::Tensor logits = output->elements()[0].toTensor();
        torch::Tensor h_1 = output->elements()[1].toTensor();
        torch::Tensor c_1 = output->elements()[2].toTensor();

        int action = logits.argmax(1).item<int>();
        std::cout << "[RESULT] Action taken: " << action << std::endl;
        std::cout << "[RESULT] Hidden state norm: " << h_1.norm().item<float>() << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Exception during model.forward or output extraction: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "[INFO] Inference complete." << std::endl;
    return 0;
}
