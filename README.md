# PPO LSTM

Custom PPO-LSTM implementation based on the CleanRL recurrent PPO pattern, adapted for non-CNN Gymnasium environments with vector observations and discrete action spaces. The project also includes a TorchScript export path for C++ inference through the `my_agent_cpp` LibTorch example.

## What This Is

The main training script, `ppo_lstm.py`, keeps the core CleanRL-style PPO loop while replacing the CNN encoder used in Atari-style examples with an MLP encoder for flat/vector observations. The recurrent policy is:

```text
observation -> MLP encoder -> LSTM -> actor / critic heads
```

This is meant for MDP or POMDP environments where memory can help the policy resolve hidden state, delayed information, or temporal context without relying on image observations.

## Repository Layout

- `ppo_lstm.py` - training script for discrete-action Gymnasium environments.
- `ppo_lstm_inference.py` - checkpoint evaluation, hidden-state collection, PCA/t-SNE plotting, and TorchScript export.
- `my_agent_cpp/` - minimal C++ LibTorch inference example for the exported policy.
- `LSTM_analysis.txt` - notes on optional hidden-state analysis methods.
- `checkpoints/` - saved PyTorch checkpoints and exported TorchScript models.
- `runs/` - TensorBoard logs.
- `plots/` - generated hidden-state visualizations.
- `Older_Version/` - earlier experiments and implementation variants.

## Setup

Create and activate a Python environment, then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pinned environment currently uses PyTorch `2.9.1+cu130`. CPU execution is also supported by setting `--cuda False` in the training script.

## Training

Run training from the project root:

```powershell
python ppo_lstm.py --env-id CartPole-v1 --total-timesteps 500000 --num-envs 8
```

Useful options:

```powershell
python ppo_lstm.py --help
```

Common arguments include:

- `--env-id` - Gymnasium environment id.
- `--total-timesteps` - total environment steps.
- `--num-envs` - number of parallel environments.
- `--num-steps` - rollout length per environment.
- `--capture-video` - save evaluation video for supported environments.
- `--cuda False` - force CPU training.
- `--track` - enable Weights & Biases logging.

Checkpoints are saved under `checkpoints/` every 100k global steps and once more at the end of training. TensorBoard metrics are written under `runs/`.

```powershell
tensorboard --logdir runs
```

or run:

```powershell
.\RunTensorBoard.bat
```

## Evaluating and Exporting

`ppo_lstm_inference.py` loads a checkpoint, evaluates the agent, optionally visualizes LSTM embeddings, and exports a TorchScript model for C++ inference.

Before running it, edit the defaults in `EvalArgs`:

- `env_id`
- `run_name`
- `checkpoint_dir`
- `num_eval_episodes`
- `analyze_embeddings`

Then run:

```powershell
python ppo_lstm_inference.py
```

The script exports:

```text
checkpoints/agent_script.pt
```

The exported TorchScript wrapper returns:

```text
logits, new_hidden_state, new_cell_state
```

The C++ side can take `argmax(logits)` for greedy action selection or apply its own sampling logic.

## C++ Inference

The `my_agent_cpp` folder contains a small LibTorch executable that loads `agent_script.pt`, creates dummy observation and LSTM state tensors, runs a forward pass, and prints the selected action and hidden-state norm.

Requirements:

- CMake
- C++17 compiler
- LibTorch matching the PyTorch version used to export the model

The included `CMakeLists.txt` currently points to:

```text
C:/libtorch-win-shared-with-deps-2.9.1+cpu/libtorch/share/cmake/Torch
```

Update `Torch_DIR` if LibTorch is installed somewhere else.

Build from `my_agent_cpp`:

```powershell
cd my_agent_cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Place `agent_script.pt` beside the compiled executable, or update the path in `main.cpp`:

```cpp
model = torch::jit::load("agent_script.pt");
```

Then run:

```powershell
.\Release\agent_inference.exe
```

The current C++ example is configured around CartPole-style dimensions:

```cpp
const int obs_dim = 4;
const int hidden_size = 128;
const int num_layers = 1;
```

Change `obs_dim` to match the environment used for training.

## Hidden-State Analysis

The inference script can collect LSTM embeddings and map them to actions with PCA and t-SNE. This is useful for checking whether recurrent memory clusters around actions, phases of behavior, or latent state.

Current analysis support:

- collect LSTM embeddings during evaluation
- save action labels
- generate PCA and t-SNE scatter plots in `plots/`

Possible extensions described in `LSTM_analysis.txt` include:

- linear probes from hidden state to action or latent state
- CCA against accessible environment state
- saliency or integrated-gradient analysis over observation history
- memory ablation by periodically resetting LSTM state

## Notes and Limitations

- The current implementation supports discrete action spaces.
- The model expects vector observations, not image/CNN observations.
- TorchScript export is CPU-oriented for the included LibTorch example.
- `ppo_lstm_inference.py` uses dataclass defaults rather than command-line parsing, so edit `EvalArgs` before evaluation/export.
- The C++ example is intentionally minimal and should be adapted to your real environment loop.

## License

This project is open source under the MIT License.
