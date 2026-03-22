import torch
import torch.optim as optim
import numpy as np

from data_loader import DataLoader
from reward_calculator import RewardCalculator
from models import PPOAgent
import constants as CONST
import tb_logger as TB

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tmp = 0

def get_synthetic_rollout(agent, steps=CONST.TRAJECTORY_STEPS):

    global tmp

    dl = DataLoader()
    rc = RewardCalculator()

    states, actions, log_probs, rewards, values = [], [], [], [], []

    for _ in range(steps):
        state_np, observation_data = dl.get_state() # (3, 90)
        state = torch.tensor(state_np[np.newaxis, :, :], dtype=torch.float32, device=device) # [1, 3, 90]

        # print('==', state_np.shape)
        # print('==', state.shape)

        with torch.no_grad():
            mu, val = agent(state)
            std = torch.exp(agent.log_std)
            dist = torch.distributions.Normal(mu, std)

            action_raw = dist.sample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)

        action = torch.clamp(action_raw, -1, 1)

        reward = np.clip(
            rc.get_reward(action[0].cpu().numpy(), observation_data),
            a_min=-3.0,
            a_max=3.0,
        )

        # CONST.WRITER.add_histogram('reward', reward, tmp)
        TB.WRITER.add_scalar('reward', reward, tmp)


        tmp = tmp + 1

        # print(tmp)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(val)

    return (
        torch.cat(states),
        torch.cat(actions),
        torch.cat(log_probs),
        rewards,
        torch.cat(values),
    )


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    v = values.flatten()
    T = len(rewards)

    advantages = torch.zeros(T, device=device)
    last_gae = 0

    for t in reversed(range(T)):
        next_v = v[t + 1] if t + 1 < T else 0
        delta = rewards[t] + gamma * next_v - v[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae

    returns = advantages + v
    return returns.view(-1, 1), advantages.view(-1, 1)


# --- Training ---
agent = PPOAgent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=CONST.LR)

for iteration in range(700):
    all_s, all_a, all_log_p, all_r, all_v = [], [], [], [], []

    # 1. Collect rollouts
    for _ in range(8):
        s, a, lp, r, v = get_synthetic_rollout(agent)
        all_s.append(s)
        all_a.append(a)
        all_log_p.append(lp)
        all_r.append(r)
        all_v.append(v)

    # 2. Compute GAE
    final_returns, final_advantages = [], []
    for i in range(8):
        ret, adv = compute_gae(all_r[i], all_v[i], gamma=CONST.GAMMA, lam=0.95)
        final_returns.append(ret)
        final_advantages.append(adv)

    # 3. Concatenate
    s_batch = torch.cat(all_s)
    a_batch = torch.cat(all_a)
    lp_batch = torch.cat(all_log_p)
    ret_batch = torch.cat(final_returns)
    adv_batch = torch.cat(final_advantages)
    v_batch = torch.cat(all_v)

    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

    MINIBATCH_SIZE = 256
    batch_size = s_batch.size(0)

    # 4. PPO optimization
    for _ in range(CONST.EPOCHS):
        indices = torch.randperm(batch_size, device=device)

        for start in range(0, batch_size, MINIBATCH_SIZE):
            idx = indices[start:start + MINIBATCH_SIZE]

            s_mb = s_batch[idx]
            a_mb = a_batch[idx]
            lp_mb = lp_batch[idx]
            ret_mb = ret_batch[idx]
            adv_mb = adv_batch[idx]
            v_old_mb = v_batch[idx]

            mu, val = agent(s_mb)
            std = torch.exp(agent.log_std)
            dist = torch.distributions.Normal(mu, std)

            new_log_p = dist.log_prob(a_mb).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # IMPORTANT: old log probs should not get gradients
            ratio = torch.exp(new_log_p - lp_mb)

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - CONST.EPS_CLIP, 1 + CONST.EPS_CLIP) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            value_clipped = v_old_mb + torch.clamp(val - v_old_mb, -0.2, 0.2)
            value_loss1 = (val - ret_mb) ** 2
            value_loss2 = (value_clipped - ret_mb) ** 2
            critic_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            loss = actor_loss + critic_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

    total_reward = sum([sum(i) for i in all_r]) * 100.0
    print(f"Iteration {iteration} | total_reward: {total_reward:.2f}")
    TB.WRITER.add_scalar('total_reward', total_reward, iteration)
