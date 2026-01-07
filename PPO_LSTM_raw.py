import numpy as np
import torch
import torch.nn as nn
import tyro
import torch.optim as optim
import time
import gymnasium as gym
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import random
from PPO_args import Args
"""
Instead of using LSTM module, will be implementing LSTM from scratch.
Optionally could use GRU.
"""

# CUDA diagnostics
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))
    except Exception:
        pass

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_cell, self).__init__()
        #all needed inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.combined_size = self.input_size + self.hidden_size
        self.forget_gate = nn.Linear(self.combined_size, hidden_size)
        self.input_gate = nn.Linear(self.combined_size, hidden_size)
        self.candidate_gate = nn.Linear(self.combined_size, hidden_size)
        self.output_gate = nn.Linear(self.combined_size, hidden_size)

    def forward(self, i, hidden_state):
        h_prev, c_prev = hidden_state

        combined = torch.cat([i, h_prev], dim=1)

        if combined.shape[1] != self.combined_size:
            raise ValueError("combined input dimension does not match input_size + hidden_size")

        #forget gate uses binary classification (sigmoid)
        f_g = torch.sigmoid(self.forget_gate(combined))

        #input gate uses binary classification (sigmoid)
        i_g = torch.sigmoid(self.input_gate(combined))

        #candidate gate uses tanh, allows for positive and negative updates
        c_g = torch.tanh(self.candidate_gate(combined))

        #output gate uses binary classification (sigmoid)
        o_g = torch.sigmoid(self.output_gate(combined))

        #Follow LSTM rules, tensor algebra
        c_new = f_g * c_prev + i_g * c_g

        h_new = o_g * torch.tanh(c_new)

        return h_new, c_new

#process sequences
class LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = LSTM_cell(input_size, hidden_size)
    
    """
    x: (batch_size, seq_len, input_size)
    hidden_state: tuple of (h0, c0) or None
    """
    def forward(self, x, hidden_state=None):
        batch_size, seq_len, input_size = x.shape
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h , c = hidden_state

        #process embeddings
        embeddings = []

        #Process timesteps to get the embeddings (these are just stacks of hidden states, ignore cell states)
        for t in range(0, seq_len):
            #input here is
            h, c = self.lstm_cell(x[:, t, :], (h,c))
            embeddings.append(h)


        embeddings = torch.stack(embeddings, dim=1)

        return embeddings, (h, c)
"""
QR Decomposition -> returns orthonormal basis times std (scalar)
Affine transformation
"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer        

"""
Entirely based on Ray's LSTM wrapper around its RL Algorithms:
https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py
"""

class Agent_LSTM_PPO(nn.Module):
    def __init__(self, obs_dim, action_dim, lstm_hidden_size=64, dense_layers=None, continuous_actions=False):
        super(Agent_LSTM_PPO, self).__init__()
        if dense_layers is None:
            dense_layers = [128, 128]

        # obs_dim can be an int or a shape tuple
        try:
            obs_size = int(np.prod(obs_dim))
        except Exception:
            obs_size = int(obs_dim)

        # action_dim should be an int (number of discrete actions)
        self.obs_dim = obs_size
        self.action_dim = int(action_dim)
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_layers = dense_layers
        self.continuous_actions = continuous_actions

        self.lstm = LSTM_Layer(self.obs_dim, lstm_hidden_size)

        # build FC embedding net that maps LSTM hidden -> embedding
        layers = []
        in_size = lstm_hidden_size
        for out_size in dense_layers:
            layers.append(layer_init(nn.Linear(in_size, out_size)))
            layers.append(nn.ReLU())
            in_size = out_size
        
        self._embeddings_fc_net = nn.Sequential(*layers) if layers else nn.Identity()

        final_embedding_size = dense_layers[-1] if len(dense_layers) > 0 else lstm_hidden_size

        # Actor head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(final_embedding_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(final_embedding_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _compute_embeddings_and_state_out(self, batch):
        # Accept either a dict batch {'obs': tensor, 'state_in': {...}} or a raw obs tensor
        if isinstance(batch, dict):
            obs = batch['obs']
            state_in = batch.get('state_in', None)
        else:
            obs = batch
            state_in = None

        if state_in is not None:
            h = state_in['h']
            c = state_in['c']
            hidden_state = (h, c)
        else:
            hidden_state = None

        # Ensure obs has sequence dim: (batch, seq_len, obs_dim)
        squeezed = False
        if obs.ndim == 2:
            obs_seq = obs.unsqueeze(1)
            squeezed = True
        else:
            obs_seq = obs

        embeddings, (h_new, c_new) = self.lstm(obs_seq, hidden_state)

        # embeddings: (batch, seq_len, lstm_hidden_size)
        b, s, hdim = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, hdim)
        embeddings_out_flat = self._embeddings_fc_net(embeddings_flat)
        embeddings_out = embeddings_out_flat.view(b, s, -1)

        # return embeddings for the last time step and the new state
        return embeddings_out, {"h": h_new, "c": c_new}

    def get_values(self, batch, embeddings):
        # kept for compatibility but prefer get_value
        if embeddings is None:
            embeddings_out, _ = self._compute_embeddings_and_state_out(batch)
            # use last time step
            embeddings_in = embeddings_out[:, -1, :]
        else:
            embeddings_in = embeddings
        values = self.critic(embeddings_in).squeeze(-1)
        return values

    def get_value(self, obs):
        embeddings_out, _ = self._compute_embeddings_and_state_out(obs)
        embeddings_in = embeddings_out[:, -1, :]
        return self.critic(embeddings_in).squeeze(-1)

    def get_action_and_value(self, batch, action=None):
        embeddings_out, state_outs = self._compute_embeddings_and_state_out(batch)
        # use last time step
        embeddings_in = embeddings_out[:, -1, :]
        logits = self.actor(embeddings_in)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(embeddings_in).squeeze(-1)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # Use rgb_array render mode when capturing video for the first env
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")

            # episode_trigger will return True whenever we've entered a new
            # global_step bucket of size VIDEO_STATE['freq'] (i.e., every 10k steps)
            def episode_trigger(episode_id: int):
                try:
                    bucket = int(global_step) // VIDEO_STATE["freq"]
                except Exception:
                    # If global_step isn't available yet, don't record
                    return False
                if bucket > VIDEO_STATE["last_bucket"]:
                    VIDEO_STATE["last_bucket"] = bucket
                    return True
                return False

            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=episode_trigger)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

"""
Modified version of CleanRL's PPO
"""

# Global state to control periodic video recording (every N steps)
VIDEO_STATE = {"last_bucket": -1, "freq": 10000}

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Instantiate agent with correct obs/action dims
    agent = Agent_LSTM_PPO(
        envs.single_observation_space.shape,
        envs.single_action_space.n,
        lstm_hidden_size=64,
        dense_layers=[128, 128],
        continuous_actions=False,
    ).to(device)
    print("Agent device:", next(agent.parameters()).device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log episodic returns when episodes finish. Handles both
            # the newer `infos` dict with `final_info` and the older/list-style infos.
            if "final_info" in infos:
                for final_info in infos["final_info"]:
                    if final_info and "episode" in final_info:
                        print(f"global_step={global_step}, episodic_return={final_info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", final_info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", final_info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()















