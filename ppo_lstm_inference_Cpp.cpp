#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // === Load the model ===
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("checkpoints/agent_script.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }

    model.eval();

    // === Dummy input ===
    const int obs_dim = 4;      // Set to actual observation size (e.g., CartPole-v1 is 4)
    const int hidden_size = 128;
    const int num_layers = 1;

    std::vector<float> obs_vec(obs_dim, 0.5f);  // Dummy observation
    torch::Tensor obs = torch::from_blob(obs_vec.data(), {1, obs_dim}).clone();

    torch::Tensor h_0 = torch::zeros({num_layers, 1, hidden_size});
    torch::Tensor c_0 = torch::zeros({num_layers, 1, hidden_size});
    torch::Tensor done = torch::zeros({1});  // Not done

    // === Forward ===
    std::vector<torch::jit::IValue> inputs = {obs, h_0, c_0, done};
    auto output = model.forward(inputs).toTuple();

    torch::Tensor logits = output->elements()[0].toTensor();
    torch::Tensor h_1 = output->elements()[1].toTensor();
    torch::Tensor c_1 = output->elements()[2].toTensor();

    // Pick action with max logit (argmax)
    int action = logits.argmax(1).item<int>();

    std::cout << "Action taken: " << action << std::endl;
    std::cout << "Hidden state norm: " << h_1.norm().item<float>() << std::endl;

    return 0;
}
