import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

@dataclass
class EvalArgs:
    env_id: str = "CartPole-v1"
    run_name: str = "CartPole-v1__ppo_lstm__1__1769466375_step1400000.pt"
    checkpoint_dir: str = "checkpoints"
    seed: int = 1
    cuda: bool = True

    num_eval_episodes: int = 50
    analyze_embeddings: bool = True
    save_plots: bool = True
    plot_dir: str = "plots"


def make_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.n
        hidden_dim = self.obs_dim * self.action_dim

        self.network = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 512)),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_states(self, x, lstm_state, done):
        x = self.network(x)
        x = x.unsqueeze(0)  # seq_len = 1
        # Reset hidden state where episodes ended
        done = done.view(1, -1, 1)
        h = (1.0 - done) * lstm_state[0]
        c = (1.0 - done) * lstm_state[1]
        output, lstm_state = self.lstm(x, (h, c))
        return output.squeeze(0), lstm_state

    def get_action(self, x, lstm_state, done):
        embedding, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(embedding)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, embedding, lstm_state

def load_latest_checkpoint(agent, run_name, checkpoint_dir, device):
    files = [f for f in os.listdir(checkpoint_dir) if run_name in f]
    if not files:
        raise FileNotFoundError("No checkpoints found.")
    latest = sorted(files)[-1]
    path = os.path.join(checkpoint_dir, latest)
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print(f"Loaded checkpoint: {latest}")
    return checkpoint.get("global_step", 0)


def evaluate(agent, env, args, device):
    agent.eval()
    embeddings = []
    actions = []
    returns = []

    obs, _ = env.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
    done = torch.zeros(1).to(device)
    lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
    )

    for ep in range(args.num_eval_episodes):
        ep_return = 0
        ep_embeddings = []
        ep_actions = []

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        done = torch.zeros(1).to(device)
        lstm_state = (
            torch.zeros_like(lstm_state[0]),
            torch.zeros_like(lstm_state[1]),
        )

        while True:
            with torch.no_grad():
                action, embedding, lstm_state = agent.get_action(obs, lstm_state, done)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            done = torch.tensor([terminated or truncated], dtype=torch.float32).to(device)

            ep_return += reward
            ep_embeddings.append(embedding.cpu().numpy())
            ep_actions.append(action.item())

            if done.item():
                break

        returns.append(ep_return)
        embeddings.extend(ep_embeddings)
        actions.extend(ep_actions)

        print(f"Episode {ep+1}/{args.num_eval_episodes} | Return: {ep_return:.2f}")

    return np.array(embeddings), np.array(actions), returns

class TorchScriptWrapper(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.network = agent.network
        self.lstm = agent.lstm
        self.actor = agent.actor

    def forward(self, obs, lstm_h, lstm_c, done):
        # Input shapes:
        # - obs: (batch, obs_dim)
        # - lstm_h, lstm_c: (num_layers=1, batch, hidden_size=128)
        # - done: (batch,)

        x = self.network(obs)
        x = x.unsqueeze(0)  # seq_len = 1

        done = done.view(1, -1, 1)
        lstm_h = (1.0 - done) * lstm_h
        lstm_c = (1.0 - done) * lstm_c
        output, (new_h, new_c) = self.lstm(x, (lstm_h, lstm_c))

        embedding = output.squeeze(0)
        logits = self.actor(embedding)
        return logits, new_h, new_c

def export_torchscript(agent, device, output_path="checkpoints/agent_script.pt"):
    wrapper = TorchScriptWrapper(agent).to(device)
    wrapper.eval()

    obs = torch.randn(1, agent.obs_dim).to(device)
    h = torch.zeros(1, 1, 128).to(device)
    c = torch.zeros(1, 1, 128).to(device)
    done = torch.zeros(1).to(device)

    scripted = torch.jit.trace(wrapper, (obs, h, c, done))
    scripted.save(output_path)
    print(f"TorchScript model saved to: {output_path}")


def visualize_embeddings(embeddings, actions, args):
    os.makedirs(args.plot_dir, exist_ok=True)
    #Flatten embeddings:
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    actions = np.array(actions).astype(int)

    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
    tsne_result = tsne.fit_transform(embeddings)

    cmap = plt.cm.get_cmap('tab10', 2)
    # Plot PCA
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("PCA - LSTM Embeddings by Action")
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=actions, cmap=cmap, s=5)
    plt.colorbar(label="Action")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Plot t-SNE
    plt.subplot(1, 2, 2)
    plt.title("t-SNE - LSTM Embeddings by Action")
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=actions, cmap=cmap, s=5)
    plt.colorbar(label="Action")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.tight_layout()
    if args.save_plots:
        path = os.path.join(args.plot_dir, f"{args.run_name}_embeddings.png")
        plt.savefig(path)
        print(f"Saved plot to {path}")
    plt.show()

def compute_saliency(agent, obs_seq, lstm_state, device):
    agent.eval()
    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device).unsqueeze(1)  # (T, 1, obs_dim)
    obs_seq.requires_grad_(True)

    done_seq = torch.zeros((len(obs_seq), 1)).to(device)

    embedding = agent.network(obs_seq)
    outputs, _ = agent.lstm(embedding, lstm_state)
    final_embed = outputs[-1]  # last time step
    logits = agent.actor(final_embed)
    selected_logit = logits.squeeze(0).max()  # max logit

    selected_logit.backward()
    saliency = obs_seq.grad.abs().squeeze(1).cpu().numpy()  # (T, obs_dim)

    return saliency



if __name__ == "__main__":
    args = EvalArgs()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.seed)
    agent = Agent(env).to(device)

    _ = load_latest_checkpoint(agent, args.run_name, args.checkpoint_dir, device)

    embeddings, actions, returns = evaluate(agent, env, args, device)

    print(f"\n Avg Return over {args.num_eval_episodes} episodes: {np.mean(returns):.2f}")
    print(f" Return Std Dev: {np.std(returns):.2f}")

    #Torch script support cpu for now..
    export_torchscript(agent, 'cpu')

    if args.analyze_embeddings:
        visualize_embeddings(embeddings, actions, args)
