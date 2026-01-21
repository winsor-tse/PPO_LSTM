import ale_py
import numpy as np
import torch
import torch.nn as nn
import tyro
import torch.optim as optim
import time
import gymnasium as gym
from gymnasium.envs.registration import register
from torch.distributions.categorical import Categorical
import random
from PPO_args import Args
from torch.utils.tensorboard import SummaryWriter
from LSTM_from_scratch import LSTM_Layer

"""
Still does not work yet:
(LSTM) state and sequences are handled during training is wrong.
1) LSTM Masking During Training Is Incomplete
2) recurrent minibatching is incorrect
3) LSTM state used at training time is wrong
4) The LSTM interface mismatch
"""


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
#Supports Discrete Actions with Non CNN based Atari games
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

    def _compute_embeddings_and_state_out(self, batch, mask_done=True, zero_state_at_done=False):
        # Accept either a dict batch {'obs': tensor, 'state_in': {...}} or a raw obs tensor
        if isinstance(batch, dict):
            obs = batch['obs']
            state_in = batch.get('state_in', None)
            dones = batch.get('dones', None)
        else:
            obs = batch
            state_in = None
            dones = None

        hidden_state = None
        # LSTM Masking: Reset hidden state when environments are done (prevents information leakage)
        if state_in is not None:
            h, c = state_in["h"], state_in["c"]
            if dones is not None and mask_done:
                # Mask out finished envs: h, c shape [batch, hidden_size] or [1, batch, hidden_size]
                if h.ndim == 3:  # [1, num_envs, hidden_size]
                    done_mask = dones.float().view(1, -1, 1)
                else:  # [batch, hidden_size]
                    done_mask = dones.float().view(-1, 1)
                h = h * (1.0 - done_mask)
                c = c * (1.0 - done_mask)
            hidden_state = (h, c)

        # For pixel based or image based obs
        # Flatten observation if needed: (batch, *obs_shape) -> (batch, obs_dim)
        if obs.ndim > 2:
            batch_size = obs.shape[0]
            obs = obs.reshape(batch_size, -1)
        
        # Ensure obs has sequence dim: (batch, seq_len, obs_dim)
        squeezed = False
        if obs.ndim == 2:
            obs_seq = obs.unsqueeze(1)
            squeezed = True
        else:
            obs_seq = obs

        # Zero LSTM state mid-sequence: reset state when done=True within sequence
        embeddings, (h_new, c_new) = self._lstm_forward_with_done_masking(
            obs_seq, hidden_state, dones, zero_state_at_done
        )

        # embeddings: (batch, seq_len, lstm_hidden_size)
        b, s, hdim = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, hdim)
        embeddings_out_flat = self._embeddings_fc_net(embeddings_flat)
        embeddings_out = embeddings_out_flat.view(b, s, -1)

        # return embeddings for the last time step and the new state
        return embeddings_out, {"h": h_new, "c": c_new}
    
    def _lstm_forward_with_done_masking(self, x, hidden_state, dones, zero_state_at_done=False):
        """
        Forward pass through LSTM with optional masking of states at done boundaries.
        
        Args:
            x: (batch_size, seq_len, input_size)
            hidden_state: tuple of (h, c) or None
            dones: (batch_size, seq_len) or None - True where episode is done
            zero_state_at_done: if True, zero state when done=True within sequence
        
        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
            (h_new, c_new): final hidden and cell states
        """
        batch_size, seq_len, input_size = x.shape
        device = x.device
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.lstm_hidden_size, device=device)
            c = torch.zeros(batch_size, self.lstm_hidden_size, device=device)
        else:
            h, c = hidden_state
            # Ensure correct shape
            if h.ndim == 3:
                h = h.squeeze(0)
            if c.ndim == 3:
                c = c.squeeze(0)

        embeddings = []
        
        for t in range(seq_len):
            # Standard LSTM step
            h, c = self.lstm.lstm_cell(x[:, t, :], (h, c))
            embeddings.append(h)
            
            # Zero LSTM state mid-sequence if episode is done
            if zero_state_at_done and dones is not None:
                # dones[t] indicates if episode ended at this timestep
                done_mask = dones[:, t].float().view(-1, 1)
                h = h * (1.0 - done_mask)
                c = c * (1.0 - done_mask)
        
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings, (h.unsqueeze(0), c.unsqueeze(0))

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


def make_recurrent_minibatches(data, initial_states, num_minibatches, num_steps, num_envs):
    """
    Yields recurrent minibatches of length = num_steps with LSTM states.
    """

    # reshape data: [T, N, ...]
    obs = data['obs']
    actions = data['actions']
    dones = data['dones']
    logprobs = data['logprobs']
    advantages = data['advantages']
    returns = data['returns']
    values = data['values']

    env_inds = np.arange(num_envs)
    np.random.shuffle(env_inds)
    envs_per_batch = num_envs // num_minibatches

    minibatches = []
    for start in range(0, num_envs, envs_per_batch):
        sub_env_inds = env_inds[start:start+envs_per_batch]

        batch_obs = obs[:, sub_env_inds]
        batch_actions = actions[:, sub_env_inds]
        batch_dones = dones[:, sub_env_inds]
        batch_logprobs = logprobs[:, sub_env_inds]
        batch_adv = advantages[:, sub_env_inds]
        batch_ret = returns[:, sub_env_inds]
        batch_val  = values[:, sub_env_inds]

        # initial states for this group
        h_in = initial_states['h'][:, sub_env_inds]
        c_in = initial_states['c'][:, sub_env_inds]

        minibatches.append({
            'obs': batch_obs,
            'actions': batch_actions,
            'dones': batch_dones,
            'logprobs': batch_logprobs,
            'advantages': batch_adv,
            'returns': batch_ret,
            'values': batch_val,
            'lstm_h': h_in,
            'lstm_c': c_in,
        })

    return minibatches



def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # Import ale_py here to ensure ALE environments are registered
        try:
            import ale_py
        except ImportError:
            pass
        
        # Use rgb_array render mode when capturing video for the first env
        if capture_video and idx == 0:
            # Check if it's an Atari environment
            if "NoFrameskip" in env_id or "ALE/" in env_id:
                env = gym.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id, render_mode="rgb_array", continuous=False)

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
            # Check if it's an Atari environment
            if "NoFrameskip" in env_id or "ALE/" in env_id:
                env = gym.make(env_id, obs_type="ram")
                print("Making RAM version")
            else:
                env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
    
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

    #from Sync -> Async
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Instantiate agent with correct obs/action dims
    agent = Agent_LSTM_PPO(
        envs.single_observation_space.shape,
        envs.single_action_space.n,
        lstm_hidden_size=args.num_lstm_hidden_size,
        dense_layers=[128, 128],
        continuous_actions=False,
    ).to(device)
    print("Agent device:", next(agent.parameters()).device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    

    #TODO: These need top be stored by time step?
    """
    obs[t], actions[t], dones[t],
    logprobs[t], values[t],
    lstm_h_state[t], lstm_c_state[t]
    """
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Initialize LSTM state per num steps and per num envs
    lstm_h_state = torch.zeros(args.num_steps, args.num_envs, args.num_lstm_hidden_size, device=device)
    lstm_c_state = torch.zeros_like(lstm_h_state)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # per-environment episodic tracking for reliable logging
    ep_returns = np.zeros(args.num_envs, dtype=float)
    ep_lengths = np.zeros(args.num_envs, dtype=int)

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

            # ALGO LOGIC: action logic with per-environment LSTM state
            with torch.no_grad():
                # Create batch dict with current obs and state
                batch_dict = {
                    'obs': next_obs,
                    'state_in': {'h': lstm_h_state[step].unsqueeze(0), 'c': lstm_c_state[step].unsqueeze(0)},
                    'dones': next_done
                }
                action, logprob, _, value = agent.get_action_and_value(batch_dict)
                values[step] = value.flatten()
                
                # Update LSTM state (we need to capture it after the forward pass)
                embeddings_out, state_out = agent._compute_embeddings_and_state_out(batch_dict, mask_done=True, zero_state_at_done=False)
                lstm_h_state[step] = state_out['h'].squeeze(0)
                lstm_c_state[step] = state_out['c'].squeeze(0)
                
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            # update per-env episodic counters using numpy reward
            try:
                reward_arr = np.array(reward)
            except Exception:
                reward_arr = np.asarray(reward)
            ep_returns += reward_arr
            ep_lengths += 1

            # for any envs that finished this step, log their episodic returns
            finished = np.where(next_done)[0]
            for i in finished:
                r = float(ep_returns[i])
                l = int(ep_lengths[i])
                print(f"global_step={global_step}, episodic_return={r}")
                writer.add_scalar("charts/episodic_return", r, global_step)
                writer.add_scalar("charts/episodic_length", l, global_step)
                # reset counters for that env
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

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

        minibatches = make_recurrent_minibatches()

        for mb in minibatches:
            # mb shapes:
            # obs:      [T, B, obs_dim]
            # actions:  [T, B]
            # dones:    [T, B]

            T, B = mb["obs"].shape[:2]

            batch_dict = {
                "obs": mb["obs"],
                "state_in": {
                    "h": mb["lstm_h"],  # [1, B, H]
                    "c": mb["lstm_c"],
                },
                "dones": mb["dones"],
            }

            # 1) UNROLL LSTM OVER TIME
            embeddings, _ = agent._compute_embeddings_and_state_out(
                batch_dict,
                mask_done=True,
                zero_state_at_done=True,
            )
            # embeddings: [T, B, hidden]
            flat_embeddings = embeddings.reshape(T * B, -1)

            # 2) Policy + value from same embeddings
            logits = agent.actor(flat_embeddings)
            dist = Categorical(logits=logits)

            flat_actions = mb["actions"].reshape(-1)
            flat_logprobs = mb["logprobs"].reshape(-1)
            flat_advantages = mb["advantages"].reshape(-1)
            flat_returns = mb["returns"].reshape(-1)
            flat_values = mb["values"].reshape(-1)

            newlogprob = dist.log_prob(flat_actions)
            entropy = dist.entropy()

            ratio = (newlogprob - flat_logprobs).exp()

            # PPO loss
            pg_loss1 = -flat_advantages * ratio
            pg_loss2 = -flat_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = agent.critic(flat_embeddings).squeeze(-1)
            v_loss = 0.5 * ((newvalue - flat_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

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
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Log LSTM embedding statistics once per iteration
        with torch.no_grad():
            batch_dict = {
                'obs': b_obs,
                'dones': b_dones
            }
            embeddings_out, _ = agent._compute_embeddings_and_state_out(batch_dict, mask_done=True, zero_state_at_done=False)
            embeddings_in = embeddings_out[:, -1, :]  # last timestep
            emb_norm = torch.norm(embeddings_in, dim=1).mean()
            emb_mean = embeddings_in.mean()
            emb_std = embeddings_in.std()
            writer.add_scalar("embeddings/norm", emb_norm.item(), global_step)
            writer.add_scalar("embeddings/mean", emb_mean.item(), global_step)
            writer.add_scalar("embeddings/std", emb_std.item(), global_step)

    envs.close()
    writer.close()