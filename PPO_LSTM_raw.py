import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from PPO_args import Args
"""
Instead of using LSTM module, will be implementing LSTM from scratch.

Optionally could use GRU.

"""

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_cell, self).__init__()
        #all needed inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = self.input_size + self.hidden_size
        self.forget_gate = nn.Linear(combined_size, hidden_size)
        self.input_gate = nn.Linear(combined_size, hidden_size)
        self.candidate_gate = nn.Linear(combined_size, hidden_size)
        self.output_gate = nn.Linear(combined_size, hidden_size)

    def forward(self, i, hidden_state):
        h_prev, c_prev = hidden_state

        combined = torch.cat([i, h_prev], dim=1)

        if len(combined) != combined_size:
            #throw error combined size does not match the catted input_size + hidden_size
            return

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
        batch_size, seq_len, input_size = X.shape
        if hidden_state == None:
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


class Agent_LSTM_PPO(nn.Module):
    def __init__(self, obs_dim, action_dim, lstm_hidden_size, dense_layers: list[128, 128], continous_actions):
        super(Agent_LSTM_PPO, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_layers = dense_layers
        self.continous_actions = continous_actions
        self.lstm = LSTM_Layer(obs_dim, lstm_hidden_size)

        #Following Code will use FCNET to compute embeddings
        #Embeddings will be feed through this

        layers = []
        #outsize defaulted as 128
        in_size = lstm_hidden_size
        #size of 2
        for out_size in dense_layers:
            layers.append(layer_init(nn.Linear(in_size, out_size)))
            layers.append(nn.RELU())
            in_size = out_size
        # REUL(RELU(lstm_hidden_size x out_size),  out_size) -> outsize
        self._embeddings_fc_net = nn.Sequential(*layers)
        #Actor is the Policy head P(s,a)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )
        #critic only ouputs a Value(s)
        #istead of taking in envs.single_observation_space -> take in lstm_hidden size
        self.critic = nn.Sequential(
            layer_init(lstm_hidden_size, 64),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )


    def _compute_embeddings_and_state_out(self, batch):
        obs = batch['obs']
        state_in = batch.get("state_in", None)

        if state_in is not None:
            h = state_in['h']
            c = state_out['c']
            hidden_state = (h,c)
        else:
            hidden_state = None
        
        embeddings, (h_new, c_new) = self.lstm(obs, hidden_state)
        embeddings_out = self._embeddings_fc_net(embeddings)
        return embeddings_out, {"h": h_new, "c": c_new}

    def forward(self, batch):
        embeddings, state_outs = self._compute_embeddings_and_state_out(batch)
        action_logits = self.actor(embeddings)
        return action_logits

    def get_values(self, batch, embeddings):
        if embeddings is None:
            embeddings, state_outs = self._compute_embeddings_and_state_out(batch)
        values = self.critic(embeddings).squeeze(-1)
        return values

    def get_action_and_value(self, batch, action=None):
        embeddings, state_outs = self._compute_embeddings_and_state_out(batch)
        logits = self.actor(embeddings)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(embeddings).squeeze(-1)



"""
Modified version of CleanRL's PPO
"""
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent_LSTM_PPO(envs).to(device)
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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















