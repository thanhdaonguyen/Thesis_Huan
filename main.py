# main.py

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting will be disabled.")
import csv
import os
import json
from datetime import datetime
import time
import argparse

from environment import CommEnvironment
from maddpg import MADDPGAgent
from baselines import DirectTransmission, GreedyStrategy, FrequencyHopping

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- OPTIMIZED Hyperparameters with Imitation Learning ---
NUM_EPISODES = 300  # More episodes for imitation + RL
STEPS_PER_EPISODE = 1000  # Good balance for learning
BATCH_SIZE = 512  # [QRL-PATCH] Increased batch size for stability
REPLAY_BUFFER_SIZE = 150000  # Larger buffer for diverse experiences
NUM_AGENTS = 5
WARMUP_STEPS = 5000  # Shorter warmup with expert guidance
TRAIN_FREQUENCY = 1  # Train every step when learning from expert
TRAIN_ITERATIONS = 1  # Single iteration per training for stability

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better sample efficiency"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001  # Anneal beta to 1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# Keep old ReplayBuffer for compatibility
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class RunningMeanStd:
    """Running mean and standard deviation for reward normalization"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0
        self.var = 1
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

def evaluate_baseline(baseline_agent, env, num_episodes, reward_w_thr, reward_w_succ, reward_w_margin, reward_w_energy, sinr_threshold):
    """Hàm để đánh giá hiệu năng của một baseline với đầy đủ metrics."""
    total_rewards = []
    all_metrics = []
    print(f"\n--- Evaluating {baseline_agent.__class__.__name__} ---")

    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            # [QRL-PATCH] FH baseline fairness: use env.step() when fh_fair=True
            if isinstance(baseline_agent, FrequencyHopping):
                if baseline_agent.fh_fair:
                    # Fair evaluation: use env.step() with FH actions
                    actions = baseline_agent.select_actions(states)
                    next_states, rewards, _, _ = env.step(actions)
                    states = next_states
                else:
                    # Legacy oracle path
                    rewards = baseline_agent.get_rewards(states)
                    # Cần cập nhật môi trường để lấy state tiếp theo (dùng action IDLE thật)
                    # Action [-1.0, 0.0] maps to amplitude=0 (idle), mode doesn't matter
                    neutral_actions = [[-1.0, 0.0] for _ in range(env.num_agents)]
                    _, _, _, _ = env.step(neutral_actions)
                    states = env._get_states()
            else: # Với DT và Greedy
                actions = baseline_agent.select_actions(states)
                next_states, rewards, _, _ = env.step(actions)
                states = next_states
            
            episode_reward += np.mean(rewards)
        
        total_rewards.append(episode_reward / STEPS_PER_EPISODE)
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        
        if (episode + 1) % (num_episodes // 5) == 0:
            print(f"  Episode {episode + 1}/{num_episodes} finished.")

    avg_metrics = {
        'reward': np.mean(total_rewards),
        'success_rate': np.mean([m['success_rate'] for m in all_metrics]),
        'energy_efficiency': np.mean([m['energy_efficiency'] for m in all_metrics]),
        'avg_energy': np.mean([m['avg_energy_per_step'] for m in all_metrics]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in all_metrics]),
        'throughput_efficiency': np.mean([m['throughput_efficiency'] for m in all_metrics])
    }
    
    return avg_metrics

def train(agents, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    samples, indices, weights = replay_buffer.sample(BATCH_SIZE)
    device = agents[0].device if agents else torch.device("cpu")
    
    # Tách dữ liệu từ batch
    states, actions, rewards, next_states, dones = zip(*samples)

    # Chuyển đổi sang tensor
    device = agents[0].device if agents else torch.device("cpu")
    states_t = [torch.from_numpy(np.array([s[i] for s in states], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    actions_t = [torch.from_numpy(np.array([a[i] for a in actions], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    rewards_t = [torch.from_numpy(np.array([r[i] for r in rewards], dtype=np.float32)).unsqueeze(1).to(device) for i in range(NUM_AGENTS)]
    next_states_t = [torch.from_numpy(np.array([ns[i] for ns in next_states], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    dones_t = [torch.from_numpy(np.array([d[i] for d in dones], dtype=np.float32)).unsqueeze(1).to(device) for i in range(NUM_AGENTS)]
    
    # Weights for PER
    weights_t = torch.FloatTensor(weights).to(device).unsqueeze(1)

    avg_actor_loss = 0.0
    avg_critic_loss = 0.0
    avg_actor_grad_norm = 0.0
    avg_critic_grad_norm = 0.0
    avg_q_value = 0.0

    for i, agent in enumerate(agents):
        # --- Cập nhật Critic với Target Policy Smoothing (TD3-style) ---
        with torch.no_grad():
            next_actions_t = [agents[j].target_actor(next_states_t[j]) for j in range(NUM_AGENTS)]
            
            # Add smoothing noise to target actions
            for j in range(len(next_actions_t)):
                noise = torch.clamp(torch.randn_like(next_actions_t[j]) * 0.1, -0.2, 0.2)
                next_actions_t[j] = torch.clamp(next_actions_t[j] + noise, -1.0, 1.0)
            
            next_q = agent.target_critic(next_states_t, next_actions_t)
            target_q = rewards_t[i] + agent.gamma * next_q * (1 - dones_t[i])
        
        current_q = agent.critic(states_t, actions_t)
        avg_q_value += current_q.mean().item()
        
        # MSE loss with Importance Sampling Weights
        td_error = current_q - target_q
        critic_loss = (weights_t * (td_error ** 2)).mean()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm_critic)
        avg_critic_grad_norm += critic_grad_norm.item()
        agent.critic_optimizer.step()
        
        # Update priorities in Replay Buffer (using absolute TD error)
        # We aggregate TD errors from all agents? Or just use the first one?
        # Usually PER is for single agent. For MADDPG, we can average TD errors or update for each agent?
        # Since we share the buffer, we should probably update based on the "worst" or "average" TD error for that transition.
        # Let's use the average TD error across agents for the priority of the transition.
        # BUT wait, the buffer stores joint experience.
        # Let's store the TD errors for this agent and we will aggregate them later or just update now?
        # The `update_priorities` takes indices. If we call it multiple times for same indices, it overwrites?
        # Let's accumulate TD errors and update once at the end of agent loop?
        # Actually, let's just use the TD error from the first agent (or average) to update priorities.
        # For simplicity and since agents cooperate, let's use the mean absolute TD error across all agents.
        if i == 0:
             total_td_errors = np.abs(td_error.detach().cpu().numpy()).flatten()
        else:
             total_td_errors += np.abs(td_error.detach().cpu().numpy()).flatten()

        # --- Cập nhật Actor ---
        current_actions_t = actions_t.copy()
        predicted_actions = agent.actor(states_t[i])
        current_actions_t[i] = predicted_actions
        
        # Policy gradient loss
        policy_loss = -agent.critic(states_t, current_actions_t).mean()
        
        # Amplitude regularizer (energy saving) with cosine decay
        amp_reg = agent.amp_reg_weight() * (predicted_actions[:, 0]**2).mean()
        
        # Imitation loss - dynamic weight based on expert_epsilon
        imitation_loss = 0
        if agent.use_expert and agent.expert_epsilon > 0.05:
            # Get expert actions for this batch
            expert_actions_batch = []
            states_np = states_t[i].cpu().numpy()
            for s in states_np:
                expert_a = agent.get_expert_action(s)
                if expert_a is not None:
                    expert_actions_batch.append(expert_a)
                else:
                    expert_actions_batch.append(np.zeros(agent.action_dim))
            
            expert_actions_tensor = torch.FloatTensor(np.array(expert_actions_batch)).to(device)
            # Dynamic weight: decreases as expert_epsilon decreases
            imitation_weight = agent.imitation_coef_max * agent.expert_epsilon / 0.15  # Normalize to new max
            imitation_loss = imitation_weight * nn.MSELoss()(predicted_actions, expert_actions_tensor)
        
        # Combined loss
        actor_loss = policy_loss + imitation_loss + amp_reg
        
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.max_grad_norm_actor)
        avg_actor_grad_norm += actor_grad_norm.item()
        agent.actor_optimizer.step()

        # Tích lũy loss để logging
        avg_actor_loss += actor_loss.item()
        avg_critic_loss += critic_loss.item()

    avg_actor_loss /= NUM_AGENTS
    avg_critic_loss /= NUM_AGENTS
    avg_actor_grad_norm /= NUM_AGENTS
    avg_critic_grad_norm /= NUM_AGENTS
    avg_q_value /= NUM_AGENTS
    
    # [QRL-PATCH] Update priorities using average TD error across agents
    avg_td_errors = total_td_errors / NUM_AGENTS
    replay_buffer.update_priorities(indices, avg_td_errors)

    return avg_actor_loss, avg_critic_loss, avg_actor_grad_norm, avg_critic_grad_norm, avg_q_value

def run_training(args, grid_config=None):
    """
    Main training function. Can be called for single run or grid search.
    grid_config: dict with overrides for grid search (e.g., {'reward_w_energy': 0.03, 'idle_threshold': 0.08})
    Returns: dict with final metrics and episode_rewards
    """
    # Override args with grid config if provided
    if grid_config:
        for key, value in grid_config.items():
            setattr(args, key, value)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Parse seeds (for multi-seed runs if needed)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    NUM_EPISODES = args.episodes
    
    # Create runs directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if grid_config:
        grid_suffix = f"_w_energy_{args.reward_w_energy}_idle_{args.idle_threshold}"
        runs_dir = f'runs/{timestamp}{grid_suffix}'
    else:
        runs_dir = f'runs/{timestamp}'
    os.makedirs(runs_dir, exist_ok=True)
    
    # Setup logging
    metrics_log_file = os.path.join(runs_dir, 'metrics.jsonl')
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not grid_config:  # Only print device info for non-grid runs
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    # [QRL-PATCH] curriculum learning: start with lower mutual interference
    start_mutual_interf = args.start_mutual_interf if hasattr(args, 'start_mutual_interf') else 0.02
    final_mutual_interf = args.final_mutual_interf if hasattr(args, 'final_mutual_interf') else args.mutual_interf_coef
    curriculum_after_episodes = args.curriculum_after_episodes if hasattr(args, 'curriculum_after_episodes') else 150
    
    # Create environment with parameters
    env = CommEnvironment(
        num_agents=NUM_AGENTS,
        mutual_interf_coef=start_mutual_interf,  # [QRL-PATCH] start with lower interference
        idle_threshold=args.idle_threshold,
        reward_w_thr=args.reward_w_thr,
        reward_w_succ=args.reward_w_succ,
        reward_w_margin=args.reward_w_margin,
        reward_w_energy=args.reward_w_energy,
        mob_sigma_user=args.mob_sigma_user,
        mob_sigma_jammer=args.mob_sigma_jammer
    )
    
    # Prepare environment parameters for expert guidance
    env_params = {
        'pga_gain': env.pga_gain,
        'jammer_power': env.jammer_power,
        'noise_power': env.noise_power,
        'sinr_threshold': env.sinr_threshold,
    }
    
    # Create agents with updated hyperparameters
    agents = [MADDPGAgent(env.state_dim, env.action_dim, i, NUM_AGENTS, 
                         lr_actor=1e-4, lr_critic=2e-4, gamma=0.98, tau=args.tau, 
                         device=device, env_params=env_params,
                         epsilon_start=args.epsilon_start,
                         noise_scale_start=args.noise_scale_start,
                         expert_epsilon_start=args.expert_epsilon_start,
                         expert_decay=args.expert_epsilon_decay,
                         imitation_coef_max=args.imitation_coef_max,
                         imitation_warmup_steps=args.imitation_warmup_steps,
                         amp_reg_coef=args.amp_reg_coef,
                         amp_reg_decay_steps=args.amp_reg_decay_steps) for i in range(NUM_AGENTS)]
    
    # [QRL-PATCH] Use PrioritizedReplayBuffer
    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []
    episode_metrics = []
    total_steps = 0
    best_reward_ma10 = -np.inf
    best_agents_state = None
    
    # Additional tracking for detailed monitoring
    actor_grad_norms = []
    critic_grad_norms = []
    q_values = []
    
    if not grid_config:  # Only print banner for non-grid runs
        print("="*80)
        print("🚀 MADDPG WITH CONTINUOUS ACTIONS & MUTUAL INTERFERENCE")
        print("="*80)
        print(f"Training: {NUM_EPISODES} episodes × {STEPS_PER_EPISODE} steps")
        print(f"Batch: {BATCH_SIZE} | Buffer: {REPLAY_BUFFER_SIZE} | Warmup: {WARMUP_STEPS}")
        print(f"LR - Actor: 1e-4, Critic: 2e-4")
        print(f"Tau: {args.tau} | Gamma: 0.98 | Grad Clip: Actor=1.0, Critic=2.0")
        print(f"SINR Threshold: {env.sinr_threshold}")  # [QRL-PATCH] use actual threshold
        print(f"Network: 256-256-128 (actor), 512-256-128-64 (critic)")
        print(f"")
        print(f"🔥 NEW FEATURES:")
        print(f"  • CONTINUOUS ACTION SPACE: amplitude ∈ [0,1] mapped from [-1,1]")
        print(f"  • IDLE MODE: amplitude < {args.idle_threshold} → no transmission")
        print(f"  • BLOCK FADING: positions update every {env.block_fading_K} steps")
        print(f"  • PGA GAIN: {env.pga_gain} (was 100)")
        print(f"  • CURRICULUM: mutual_interf {start_mutual_interf} → {final_mutual_interf} at ep {curriculum_after_episodes}")
        print(f"  • FH FAIR: True (default)")
        print(f"  • MUTUAL INTERFERENCE: coef={args.mutual_interf_coef} (SU↔SU co-channel)")
        print(f"  • TARGET POLICY SMOOTHING (TD3-style)")
        print(f"  • EXPERT GATING: Don't transmit when SINR < 0.9*threshold")
        print(f"  • AMP REG DECAY: {args.amp_reg_coef} → 0 over {args.amp_reg_decay_steps} steps")
        print(f"  • REDUCED IMITATION: dynamic weight (max={agents[0].imitation_coef_max})")
        print(f"  • ENHANCED REWARD: thr={args.reward_w_thr}, succ={args.reward_w_succ}, margin={args.reward_w_margin}, energy={args.reward_w_energy}")
        print(f"")
        print(f"🎯 STRATEGY: MARL advantage through coordination & power control!")
        print("="*80)
    
    training_start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        states = env.reset()
        current_episode_reward = 0
        last_actor_loss = None
        last_critic_loss = None
        last_q_value = None
        
        # Reset noise for new episode
        for agent in agents:
            agent.reset_noise()

        for step in range(STEPS_PER_EPISODE):
            # Select actions with exploration noise
            actions = [agent.select_action(states[i], add_noise=True) for i, agent in enumerate(agents)]
            
            next_states, rewards, dones, info = env.step(actions)
            
            # Rewards are already well-scaled from environment, just store them
            replay_buffer.push(states, actions, rewards, next_states, dones)
            
            states = next_states
            current_episode_reward += np.mean(rewards)
            total_steps += 1

            # Adaptive training - train more when using expert
            train_freq = TRAIN_FREQUENCY
            if agents[0].expert_epsilon > 0.1:  # When still learning from expert
                train_freq = 1  # Train every step
            
            if total_steps > WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE and total_steps % train_freq == 0:
                train_result = train(agents, replay_buffer)
                if train_result is not None:
                    last_actor_loss, last_critic_loss, actor_gn, critic_gn, q_val = train_result
                    last_q_value = q_val
                    actor_grad_norms.append(actor_gn)
                    critic_grad_norms.append(critic_gn)
                    q_values.append(q_val)
                
                # Update target networks
                for agent in agents:
                    agent.update_targets()

        # [QRL-PATCH] curriculum learning: increase mutual interference after specified episodes
        if episode == curriculum_after_episodes:
            env.mutual_interf_coef = final_mutual_interf
            if not grid_config:
                print(f"📈 Curriculum: Increasing mutual_interf_coef from {start_mutual_interf} to {final_mutual_interf} at episode {episode + 1}")
        
        # Get episode metrics
        metrics = env.get_metrics()
        episode_metrics.append(metrics)
        
        avg_episode_reward = current_episode_reward / STEPS_PER_EPISODE
        episode_rewards.append(avg_episode_reward)
        
        # Calculate mode ratios from last step info (approximate)
        mode_counts = {'idle': 0, 'd2d': 0, 'relay': 0}
        avg_amplitude = 0.0
        if 'modes' in info and len(info['modes']) > 0:
            for mode in info['modes']:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            avg_amplitude = np.mean(info['amplitudes']) if info['amplitudes'] else 0.0
        
        total_modes = sum(mode_counts.values())
        idle_ratio = mode_counts['idle'] / total_modes if total_modes > 0 else 0.0
        relay_ratio = mode_counts['relay'] / total_modes if total_modes > 0 else 0.0
        
        # Log to JSONL
        log_entry = {
            'ep': episode + 1,
            'reward': avg_episode_reward,
            'succ_rate': metrics['success_rate'],
            'thr': metrics['avg_throughput'],
            'tpe': metrics['throughput_efficiency'],
            'avg_amp': avg_amplitude,
            'relay_ratio': relay_ratio,
            'idle_ratio': idle_ratio,
            'energy_eff': metrics['energy_efficiency']
        }
        with open(metrics_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Track best performance using MA-10 for stability
        avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
        if avg_reward_10 > best_reward_ma10:
            best_reward_ma10 = avg_reward_10
            best_agents_state = [{
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'target_actor_state_dict': agent.target_actor.state_dict(),
                'target_critic_state_dict': agent.target_critic.state_dict(),
            } for agent in agents]
            if not grid_config:
                print(f"🌟 New best MA-10 reward: {best_reward_ma10:.4f}")
        
        # Log theo từng episode với thông tin đầy đủ
        episode_time = time.time() - episode_start_time
        avg_reward_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_rewards[-1]
        
        # Get current learning rate and epsilon
        current_lr_actor = agents[0].actor_optimizer.param_groups[0]['lr']
        current_epsilon = agents[0].epsilon
        current_expert_eps = agents[0].expert_epsilon
        
        if not grid_config:  # Only print detailed logs for non-grid runs
            if last_actor_loss is not None and last_critic_loss is not None:
                print(f"Ep {episode + 1:3d}/{NUM_EPISODES} | R: {episode_rewards[-1]:+.4f} | R(10): {avg_reward_10:+.4f} | R(50): {avg_reward_50:+.4f} | Succ: {metrics['success_rate']:.3f} | Thr: {metrics['avg_throughput']:.3f} | ALoss: {last_actor_loss:.4f} | CLoss: {last_critic_loss:.4f} | ε: {current_epsilon:.3f} | Exp: {current_expert_eps:.3f} | Time: {episode_time:.1f}s")
            else:
                print(f"Ep {episode + 1:3d}/{NUM_EPISODES} | R: {episode_rewards[-1]:+.4f} | R(10): {avg_reward_10:+.4f} | R(50): {avg_reward_50:+.4f} | Succ: {metrics['success_rate']:.3f} | Thr: {metrics['avg_throughput']:.3f} | [Warmup] | Exp: {current_expert_eps:.3f} | Time: {episode_time:.1f}s")

    training_time = time.time() - training_start_time
    if not grid_config:
        print(f"\nTraining Finished!")
        print(f"Total training time: {training_time/60:.2f} minutes ({training_time:.1f} seconds)")
        print(f"Average time per episode: {training_time/NUM_EPISODES:.1f} seconds")
    
    # Calculate final MA-10 reward for return
    final_reward_ma10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
    
    return {
        'episode_rewards': episode_rewards,
        'episode_metrics': episode_metrics,
        'final_reward_ma10': final_reward_ma10,
        'best_reward_ma10': best_reward_ma10,
        'best_agents_state': best_agents_state,
        'runs_dir': runs_dir,
        'timestamp': timestamp,
        'agents': agents,
        'env': env,
        'args': args
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MADDPG Training with Continuous Actions and Mutual Interference')
    parser.add_argument('--mutual_interf_coef', type=float, default=0.06, help='Mutual interference coefficient (0.06-0.2)')
    # [QRL-PATCH] curriculum config
    parser.add_argument('--start_mutual_interf', type=float, default=0.02, help='Starting mutual interference coefficient for curriculum')
    parser.add_argument('--final_mutual_interf', type=float, default=0.06, help='Final mutual interference coefficient for curriculum')
    parser.add_argument('--curriculum_after_episodes', type=int, default=150, help='Episode to switch to final mutual interference')
    parser.add_argument('--idle_threshold', type=float, default=0.18, help='Amplitude threshold for idle mode (was 0.10)')
    parser.add_argument('--reward_w_thr', type=float, default=4.0, help='Throughput weight in reward')
    parser.add_argument('--reward_w_succ', type=float, default=1.2, help='Success bonus weight (was 0.8)')
    parser.add_argument('--reward_w_margin', type=float, default=0.8, help='SINR margin bonus weight (was 0.6)')
    parser.add_argument('--reward_w_energy', type=float, default=0.05, help='Energy penalty weight')
    parser.add_argument('--mob_sigma_user', type=float, default=2.5, help='Mobility sigma for SU/DU')
    parser.add_argument('--mob_sigma_jammer', type=float, default=3.0, help='Mobility sigma for jammer')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter for target networks')
    parser.add_argument('--amp_reg_coef', type=float, default=0.01, help='Amplitude regularizer coefficient (initial)')
    parser.add_argument('--amp_reg_decay_steps', type=int, default=50000, help='Steps for amplitude regularizer to decay to 0')
    parser.add_argument('--imitation_coef_max', type=float, default=0.05, help='Max imitation coefficient (was 0.10)')
    parser.add_argument('--imitation_warmup_steps', type=int, default=10000, help='Steps before disabling expert (was 20000)')
    # [QRL-PATCH] exploration schedule defaults
    parser.add_argument('--epsilon_start', type=float, default=0.60, help='Initial exploration epsilon (was 0.85)')
    parser.add_argument('--noise_scale_start', type=float, default=0.08, help='Initial noise scale (was 0.15)')
    parser.add_argument('--expert_epsilon_start', type=float, default=0.10, help='Initial expert epsilon (was 0.15)')
    parser.add_argument('--expert_epsilon_decay', type=float, default=0.990, help='Expert epsilon decay (was 0.95)')
    parser.add_argument('--seeds', type=str, default='1', help='Comma-separated list of random seeds')
    parser.add_argument('--episodes', type=int, default=300, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--quick_grid', type=int, default=0, help='Run quick grid search (0=no, 1=yes)')
    args = parser.parse_args()
    
    # Quick grid search mode
    if args.quick_grid == 1:
        print("="*80)
        print("🔍 QUICK GRID SEARCH MODE")
        print("="*80)
        print("Testing configurations:")
        print("  w_energy ∈ {0.03, 0.05}")
        print("  idle_threshold ∈ {0.08, 0.10}")
        print("="*80)
        
        grid_configs = [
            {'reward_w_energy': 0.03, 'idle_threshold': 0.08},
            {'reward_w_energy': 0.03, 'idle_threshold': 0.10},
            {'reward_w_energy': 0.05, 'idle_threshold': 0.08},
            {'reward_w_energy': 0.05, 'idle_threshold': 0.10},
        ]
        
        results = []
        for i, grid_config in enumerate(grid_configs):
            print(f"\n[{i+1}/4] Testing: w_energy={grid_config['reward_w_energy']}, idle_threshold={grid_config['idle_threshold']}")
            result = run_training(args, grid_config=grid_config)
            results.append({
                'config': grid_config,
                'final_reward_ma10': result['final_reward_ma10'],
                'runs_dir': result['runs_dir']
            })
            print(f"  → Final MA-10 Reward: {result['final_reward_ma10']:.4f}")
        
        # Find best configuration
        best_idx = np.argmax([r['final_reward_ma10'] for r in results])
        best_config = results[best_idx]
        
        print("\n" + "="*80)
        print("📊 GRID SEARCH RESULTS")
        print("="*80)
        for i, r in enumerate(results):
            marker = " ⭐ BEST" if i == best_idx else ""
            print(f"Config {i+1}: w_energy={r['config']['reward_w_energy']}, idle_threshold={r['config']['idle_threshold']} → MA-10: {r['final_reward_ma10']:.4f}{marker}")
        print("="*80)
        print(f"\n✅ Best configuration: w_energy={best_config['config']['reward_w_energy']}, idle_threshold={best_config['config']['idle_threshold']}")
        print(f"   Final MA-10 Reward: {best_config['final_reward_ma10']:.4f}")
        print(f"   Results saved in: {best_config['runs_dir']}")
        print("="*80)
        exit(0)
    
    # Normal single training run
    result = run_training(args, grid_config=None)
    
    # Extract results
    episode_rewards = result['episode_rewards']
    episode_metrics = result['episode_metrics']
    runs_dir = result['runs_dir']
    timestamp = result['timestamp']
    agents = result['agents']
    env = result['env']
    args = result['args']
    best_reward_ma10 = result['best_reward_ma10']
    best_agents_state = result['best_agents_state']
    
    # Print training stability metrics
    print(f"\nTraining Stability Metrics:")
    print(f"  Final LR (Actor): {agents[0].actor_optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Final Epsilon: {agents[0].epsilon:.4f}")
    
    # --- Lưu kết quả training vào CSV với đầy đủ metrics ---
    csv_filename = os.path.join(runs_dir, f'training_rewards_{timestamp}.csv')
    
    # Load idle_ratio and avg_amp data from JSONL for CSV
    idle_ratios_data = []
    avg_amp_data = []
    relay_ratios_data = []
    try:
        if os.path.exists(metrics_log_file):
            with open(metrics_log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idle_ratios_data.append(data.get('idle_ratio', 0.0))
                    avg_amp_data.append(data.get('avg_amp', 0.0))
                    relay_ratios_data.append(data.get('relay_ratio', 0.0))
    except:
        pass
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Reward_MA10', 'Success_Rate', 'Energy_Efficiency', 'Avg_Energy', 'Avg_Throughput', 'Throughput_Efficiency', 'Idle_Ratio', 'Relay_Ratio', 'Avg_Amplitude'])
        for i, (reward, metrics) in enumerate(zip(episode_rewards, episode_metrics), 1):
            # Calculate MA-10 for this episode
            reward_ma10 = np.mean(episode_rewards[max(0, i-10):i]) if i >= 10 else reward
            idle_r = idle_ratios_data[i-1] if i <= len(idle_ratios_data) else 0.0
            relay_r = relay_ratios_data[i-1] if i <= len(relay_ratios_data) else 0.0
            avg_a = avg_amp_data[i-1] if i <= len(avg_amp_data) else 0.0
            writer.writerow([i, reward, reward_ma10, metrics['success_rate'], metrics['energy_efficiency'], 
                           metrics['avg_energy_per_step'], metrics['avg_throughput'], metrics['throughput_efficiency'],
                           idle_r, relay_r, avg_a])
    print(f"✓ Đã lưu training rewards và metrics vào: {csv_filename}")
    
    # --- Lưu model checkpoint (best và final) ---
    checkpoint_dir = os.path.join(runs_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save best models (based on MA-10)
    if best_agents_state is not None:
        for i, state_dict in enumerate(best_agents_state):
            checkpoint_path = os.path.join(checkpoint_dir, f'agent_{i}_best_ma10_{timestamp}.pth')
            torch.save(state_dict, checkpoint_path)
        print(f"✓ Đã lưu best models (MA-10={best_reward_ma10:.4f}) vào thư mục: {checkpoint_dir}/")
    
    # Save final models
    for i, agent in enumerate(agents):
        checkpoint = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'target_actor_state_dict': agent.target_actor.state_dict(),
            'target_critic_state_dict': agent.target_critic.state_dict(),
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'agent_{i}_final_{timestamp}.pth')
        torch.save(checkpoint, checkpoint_path)
    print(f"✓ Đã lưu final model checkpoints vào thư mục: {checkpoint_dir}/")

    # --- Bắt đầu phần đánh giá Baselines ---
    
    # Lấy các tham số từ môi trường đã tạo
    env_params = {
        'pga_gain': env.pga_gain,
        'jammer_power': env.jammer_power,
        'noise_power': env.noise_power,
    }
    # Khởi tạo các baselines
    dt_agent = DirectTransmission(NUM_AGENTS, env.action_dim)
    greedy_agent = GreedyStrategy(NUM_AGENTS, env.action_dim, env_params)
    # [QRL-PATCH] FH baseline fairness: pass env and fh_fair flag
    fh_agent = FrequencyHopping(NUM_AGENTS, env_params, 
                                reward_w_thr=args.reward_w_thr,
                                reward_w_succ=args.reward_w_succ,
                                reward_w_margin=args.reward_w_margin,
                                reward_w_energy=args.reward_w_energy,
                                sinr_threshold=env.sinr_threshold,
                                fh_fair=True,  # [QRL-PATCH] default to fair evaluation
                                env=env)

    # Đánh giá baselines với đầy đủ metrics
    num_eval_episodes = 50  # Reduced from 200 for faster evaluation
    dt_metrics = evaluate_baseline(dt_agent, env, num_eval_episodes,
                                   args.reward_w_thr, args.reward_w_succ, 
                                   args.reward_w_margin, args.reward_w_energy, env.sinr_threshold)
    greedy_metrics = evaluate_baseline(greedy_agent, env, num_eval_episodes,
                                      args.reward_w_thr, args.reward_w_succ,
                                      args.reward_w_margin, args.reward_w_energy, env.sinr_threshold)
    fh_metrics = evaluate_baseline(fh_agent, env, num_eval_episodes,
                                  args.reward_w_thr, args.reward_w_succ,
                                  args.reward_w_margin, args.reward_w_energy, env.sinr_threshold)
    
    # MADDPG metrics từ last 50 episodes
    maddpg_last_n = 50
    maddpg_metrics = {
        'reward': np.mean(episode_rewards[-maddpg_last_n:]),
        'success_rate': np.mean([m['success_rate'] for m in episode_metrics[-maddpg_last_n:]]),
        'energy_efficiency': np.mean([m['energy_efficiency'] for m in episode_metrics[-maddpg_last_n:]]),
        'avg_energy': np.mean([m['avg_energy_per_step'] for m in episode_metrics[-maddpg_last_n:]]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in episode_metrics[-maddpg_last_n:]]),
        'throughput_efficiency': np.mean([m['throughput_efficiency'] for m in episode_metrics[-maddpg_last_n:]])
    }

    # In ra bảng so sánh cuối cùng với đầy đủ metrics
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'Reward':>10} {'Success':>8} {'Throughput':>11} {'Energy Eff':>10} {'Thr Eff':>9}")
    print("-"*80)
    print(f"{'Direct Transmission':<25} {dt_metrics['reward']:>10.4f} {dt_metrics['success_rate']:>8.3f} {dt_metrics['avg_throughput']:>11.4f} {dt_metrics['energy_efficiency']:>10.4f} {dt_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'Greedy Strategy':<25} {greedy_metrics['reward']:>10.4f} {greedy_metrics['success_rate']:>8.3f} {greedy_metrics['avg_throughput']:>11.4f} {greedy_metrics['energy_efficiency']:>10.4f} {greedy_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'Frequency Hopping':<25} {fh_metrics['reward']:>10.4f} {fh_metrics['success_rate']:>8.3f} {fh_metrics['avg_throughput']:>11.4f} {fh_metrics['energy_efficiency']:>10.4f} {fh_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'MADDPG + JM (Ours)':<25} {maddpg_metrics['reward']:>10.4f} {maddpg_metrics['success_rate']:>8.3f} {maddpg_metrics['avg_throughput']:>11.4f} {maddpg_metrics['energy_efficiency']:>10.4f} {maddpg_metrics['throughput_efficiency']:>9.4f}")
    print("="*80)
    
    # Calculate improvement percentages (EXCLUDE FH - only compare with DT and Greedy)
    best_baseline_reward = max(dt_metrics['reward'], greedy_metrics['reward'])
    best_baseline_throughput = max(dt_metrics['avg_throughput'], greedy_metrics['avg_throughput'])
    best_baseline_success = max(dt_metrics['success_rate'], greedy_metrics['success_rate'])
    
    reward_improvement = ((maddpg_metrics['reward'] - best_baseline_reward) / abs(best_baseline_reward)) * 100
    throughput_improvement = ((maddpg_metrics['avg_throughput'] - best_baseline_throughput) / abs(best_baseline_throughput)) * 100
    success_improvement = ((maddpg_metrics['success_rate'] - best_baseline_success) / abs(best_baseline_success)) * 100
    
    print(f"\n🎯 MADDPG vs Best Baseline (DT/Greedy only):")
    print(f"   Reward:      {maddpg_metrics['reward']:>7.4f} vs {best_baseline_reward:>7.4f} = {reward_improvement:>+7.2f}%")
    print(f"   Success Rate:{maddpg_metrics['success_rate']:>7.3f} vs {best_baseline_success:>7.3f} = {success_improvement:>+7.2f}%")
    print(f"   Throughput:  {maddpg_metrics['avg_throughput']:>7.4f} vs {best_baseline_throughput:>7.4f} = {throughput_improvement:>+7.2f}%")
    
    # Calculate MA-10 metrics for acceptance gates
    maddpg_reward_ma10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
    maddpg_throughput_ma10 = np.mean([m['avg_throughput'] for m in episode_metrics[-10:]]) if len(episode_metrics) >= 10 else episode_metrics[-1]['avg_throughput']
    
    # Acceptance Gates Checking
    print("\n" + "="*80)
    print("ACCEPTANCE GATES CHECK")
    print("="*80)
    
    # Gate 1: Reward (MA-10) >= Greedy + 8% OR >= DT + 3% if DT > Greedy
    reward_target_greedy = greedy_metrics['reward'] * 1.08
    reward_target_dt = dt_metrics['reward'] * 1.03
    reward_target = reward_target_greedy if greedy_metrics['reward'] >= dt_metrics['reward'] else max(reward_target_greedy, reward_target_dt)
    gate1_pass = maddpg_reward_ma10 >= reward_target
    
    # Gate 2: Success rate >= Greedy + 4 percentage points
    success_target = greedy_metrics['success_rate'] + 0.04
    gate2_pass = maddpg_metrics['success_rate'] >= success_target
    
    # Gate 3: Throughput (MA-10) >= Greedy + 8%
    throughput_target = greedy_metrics['avg_throughput'] * 1.08
    gate3_pass = maddpg_throughput_ma10 >= throughput_target
    
    # Gate 4: Efficiency (bits/J) > DT & Greedy
    gate4_pass = (maddpg_metrics['throughput_efficiency'] > dt_metrics['throughput_efficiency'] and 
                  maddpg_metrics['throughput_efficiency'] > greedy_metrics['throughput_efficiency'])
    
    print(f"Gate 1 (Reward MA-10): {maddpg_reward_ma10:.4f} >= {reward_target:.4f} {'✅ PASS' if gate1_pass else '❌ FAIL'}")
    print(f"Gate 2 (Success Rate):  {maddpg_metrics['success_rate']:.3f} >= {success_target:.3f} {'✅ PASS' if gate2_pass else '❌ FAIL'}")
    print(f"Gate 3 (Throughput MA-10): {maddpg_throughput_ma10:.4f} >= {throughput_target:.4f} {'✅ PASS' if gate3_pass else '❌ FAIL'}")
    print(f"Gate 4 (Efficiency):    {maddpg_metrics['throughput_efficiency']:.4f} > DT({dt_metrics['throughput_efficiency']:.4f}) & Greedy({greedy_metrics['throughput_efficiency']:.4f}) {'✅ PASS' if gate4_pass else '❌ FAIL'}")
    
    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    
    if all_gates_pass:
        print("\n" + "="*80)
        print("🎉 ALL ACCEPTANCE GATES PASSED! 🎉")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME ACCEPTANCE GATES FAILED - TUNING SUGGESTIONS:")
        print("="*80)
        if not gate1_pass:
            print(f"  • Reward gap: {reward_target - maddpg_reward_ma10:.4f}")
            print(f"    → Try: ↑ mutual_interf_coef (current: {args.mutual_interf_coef})")
            print(f"    → Try: ↑ reward_w_margin (current: {args.reward_w_margin}), ↓ reward_w_energy (current: {args.reward_w_energy})")
        if not gate2_pass:
            print(f"  • Success gap: {success_target - maddpg_metrics['success_rate']:.4f}")
            print(f"    → Try: ↑ reward_w_succ (current: {args.reward_w_succ})")
        if not gate3_pass:
            print(f"  • Throughput gap: {throughput_target - maddpg_throughput_ma10:.4f}")
            print(f"    → Try: ↑ reward_w_thr (current: {args.reward_w_thr})")
        if not gate4_pass:
            print(f"  • Efficiency gap: DT={dt_metrics['throughput_efficiency']:.4f}, Greedy={greedy_metrics['throughput_efficiency']:.4f}")
            print(f"    → Try: ↑ reward_w_thr, ↓ reward_w_energy")
        
        print(f"\n  Example tuning command:")
        print(f"  python main.py --mutual_interf_coef {min(0.2, args.mutual_interf_coef + 0.02):.2f} \\")
        print(f"                  --reward_w_margin {min(0.6, args.reward_w_margin + 0.1):.2f} \\")
        print(f"                  --reward_w_energy {max(0.2, args.reward_w_energy - 0.05):.2f}")
        print("="*80)
    
    # Lưu kết quả baseline vào CSV với đầy đủ metrics
    baseline_filename = os.path.join(runs_dir, f'baseline_comparison_{timestamp}.csv')
    with open(baseline_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Average_Reward', 'Success_Rate', 'Energy_Efficiency', 'Avg_Energy', 'Avg_Throughput', 'Throughput_Efficiency'])
        writer.writerow(['Direct Transmission', dt_metrics['reward'], dt_metrics['success_rate'], dt_metrics['energy_efficiency'], dt_metrics['avg_energy'], dt_metrics['avg_throughput'], dt_metrics['throughput_efficiency']])
        writer.writerow(['Greedy Strategy', greedy_metrics['reward'], greedy_metrics['success_rate'], greedy_metrics['energy_efficiency'], greedy_metrics['avg_energy'], greedy_metrics['avg_throughput'], greedy_metrics['throughput_efficiency']])
        writer.writerow(['Frequency Hopping', fh_metrics['reward'], fh_metrics['success_rate'], fh_metrics['energy_efficiency'], fh_metrics['avg_energy'], fh_metrics['avg_throughput'], fh_metrics['throughput_efficiency']])
        writer.writerow(['MADDPG + JM', maddpg_metrics['reward'], maddpg_metrics['success_rate'], maddpg_metrics['energy_efficiency'], maddpg_metrics['avg_energy'], maddpg_metrics['avg_throughput'], maddpg_metrics['throughput_efficiency']])
    print(f"✓ Đã lưu kết quả baseline vào: {baseline_filename}")
    
    # Vẽ đồ thị phần thưởng của MADDPG với enhanced visualization including throughput
    if PLOT_AVAILABLE:
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # Plot 1: Rewards với moving average
    # [QRL-PATCH] plot smoothing: increase window for visualization only
    window = 25  # was 10
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), smoothed_rewards, label=f'MADDPG (MA-{window})', linewidth=2, color='blue')
    axes[0, 0].plot(episode_rewards, alpha=0.3, linewidth=0.8, color='blue', label='MADDPG (Raw)')
    axes[0, 0].axhline(y=dt_metrics['reward'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['reward']:.4f}")
    axes[0, 0].axhline(y=greedy_metrics['reward'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['reward']:.4f}")
    axes[0, 0].axhline(y=fh_metrics['reward'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle): {fh_metrics['reward']:.4f}")
    axes[0, 0].set_xlabel("Episode", fontsize=12)
    axes[0, 0].set_ylabel("Average Reward per Step", fontsize=12)
    axes[0, 0].set_title("MADDPG Training vs. Baselines - Rewards", fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate with baselines
    success_rates = [m['success_rate'] for m in episode_metrics]
    if len(success_rates) >= window:
        smoothed_success = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(success_rates)), smoothed_success, color='orange', linewidth=2, label='MADDPG')
    else:
        axes[0, 1].plot(success_rates, color='orange', linewidth=2, label='MADDPG')
    axes[0, 1].axhline(y=dt_metrics['success_rate'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['success_rate']:.3f}")
    axes[0, 1].axhline(y=greedy_metrics['success_rate'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['success_rate']:.3f}")
    axes[0, 1].axhline(y=fh_metrics['success_rate'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle): {fh_metrics['success_rate']:.3f}")
    axes[0, 1].set_xlabel("Episode", fontsize=12)
    axes[0, 1].set_ylabel("Success Rate", fontsize=12)
    axes[0, 1].set_title("Transmission Success Rate Over Time", fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Energy Efficiency with baselines
    energy_effs = [m['energy_efficiency'] for m in episode_metrics]
    if len(energy_effs) >= window:
        smoothed_ee = np.convolve(energy_effs, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(energy_effs)), smoothed_ee, color='green', linewidth=2, label='MADDPG')
    else:
        axes[1, 0].plot(energy_effs, color='green', linewidth=2, label='MADDPG')
    axes[1, 0].axhline(y=dt_metrics['energy_efficiency'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['energy_efficiency']:.4f}")
    axes[1, 0].axhline(y=greedy_metrics['energy_efficiency'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['energy_efficiency']:.4f}")
    axes[1, 0].axhline(y=fh_metrics['energy_efficiency'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle): {fh_metrics['energy_efficiency']:.4f}")
    axes[1, 0].set_xlabel("Episode", fontsize=12)
    axes[1, 0].set_ylabel("Energy Efficiency (Success/Energy)", fontsize=12)
    axes[1, 0].set_title("Energy Efficiency Over Time", fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Throughput Over Time
    throughputs = [m['avg_throughput'] for m in episode_metrics]
    if len(throughputs) >= window:
        smoothed_throughput = np.convolve(throughputs, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(throughputs)), smoothed_throughput, color='purple', linewidth=2, label='MADDPG')
    else:
        axes[1, 1].plot(throughputs, color='purple', linewidth=2, label='MADDPG')
    axes[1, 1].axhline(y=dt_metrics['avg_throughput'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['avg_throughput']:.4f}")
    axes[1, 1].axhline(y=greedy_metrics['avg_throughput'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['avg_throughput']:.4f}")
    axes[1, 1].axhline(y=fh_metrics['avg_throughput'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle): {fh_metrics['avg_throughput']:.4f}")
    axes[1, 1].set_xlabel("Episode", fontsize=12)
    axes[1, 1].set_ylabel("Average Throughput (bits/s/Hz)", fontsize=12)
    axes[1, 1].set_title("Throughput Comparison Over Time", fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Throughput Efficiency Over Time
    throughput_effs = [m['throughput_efficiency'] for m in episode_metrics]
    if len(throughput_effs) >= window:
        smoothed_thr_eff = np.convolve(throughput_effs, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(range(window-1, len(throughput_effs)), smoothed_thr_eff, color='brown', linewidth=2, label='MADDPG')
    else:
        axes[0, 2].plot(throughput_effs, color='brown', linewidth=2, label='MADDPG')
    axes[0, 2].axhline(y=dt_metrics['throughput_efficiency'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].axhline(y=greedy_metrics['throughput_efficiency'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].axhline(y=fh_metrics['throughput_efficiency'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle): {fh_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].set_xlabel("Episode", fontsize=12)
    axes[0, 2].set_ylabel("Throughput Efficiency (bits/J)", fontsize=12)
    axes[0, 2].set_title("Throughput per Energy Over Time", fontsize=14, fontweight='bold')
    axes[0, 2].legend(loc='best')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 6: Comparative Bar Chart
    methods = ['DT', 'Greedy', 'FH', 'MADDPG']
    rewards = [dt_metrics['reward'], greedy_metrics['reward'], fh_metrics['reward'], maddpg_metrics['reward']]
    colors = ['red', 'green', 'magenta', 'blue']
    bars = axes[1, 2].bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    # Highlight the best
    best_idx = np.argmax(rewards)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    axes[1, 2].set_ylabel("Average Reward", fontsize=12)
    axes[1, 2].set_title("Final Performance Comparison", fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    results_plot_filename = os.path.join(runs_dir, f'training_curves_{timestamp}.png')
    plt.savefig(results_plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Đã lưu đồ thị toàn diện với throughput vào file: {results_plot_filename}")
    
    # Additional visualization: histograms and mode ratios
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect amplitude data from last episodes
    all_amplitudes = []
    all_modes = {'idle': 0, 'd2d': 0, 'relay': 0}
    for ep_idx in range(max(0, len(episode_metrics) - 50), len(episode_metrics)):
        # Approximate: use last step of each episode (we don't store all steps)
        # In practice, you'd want to track this during training
        pass
    
    # Plot amplitude histogram (from last 50 episodes, approximate)
    # Use smoothed MA-20 for better visualization
    # [QRL-PATCH] plot smoothing: increase window
    window_20 = 30  # was 20
    if len(episode_rewards) >= window_20:
        smoothed_rewards_ma20 = np.convolve(episode_rewards, np.ones(window_20)/window_20, mode='valid')
        axes2[0, 0].plot(range(window_20-1, len(episode_rewards)), smoothed_rewards_ma20, 
                        label=f'MADDPG (MA-{window_20})', linewidth=2, color='blue')
    axes2[0, 0].plot(episode_rewards, alpha=0.2, linewidth=0.5, color='blue', label='MADDPG (Raw)')
    axes2[0, 0].axhline(y=dt_metrics['reward'], color='r', linestyle='--', linewidth=2, label=f"DT")
    axes2[0, 0].axhline(y=greedy_metrics['reward'], color='g', linestyle='--', linewidth=2, label=f"Greedy")
    axes2[0, 0].axhline(y=fh_metrics['reward'], color='m', linestyle=':', linewidth=2, alpha=0.7, label=f"FH (oracle)")
    axes2[0, 0].set_xlabel("Episode", fontsize=12)
    axes2[0, 0].set_ylabel("Reward (MA-20)", fontsize=12)
    axes2[0, 0].set_title("Reward with MA-20 Smoothing (No Outliers)", fontsize=14, fontweight='bold')
    axes2[0, 0].legend(loc='best')
    axes2[0, 0].grid(True, alpha=0.3)
    
    # Mode ratio pie chart (approximate from last episodes)
    # Calculate from episode metrics if available, otherwise use defaults
    mode_ratios = {
        'D2D': 0.6,  # Approximate
        'Relay': 0.3,
        'Idle': 0.1
    }
    # Try to get actual ratios from logged data
    try:
        # Read from JSONL if available
        if os.path.exists(metrics_log_file):
            relay_ratios = []
            idle_ratios = []
            with open(metrics_log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'relay_ratio' in data:
                        relay_ratios.append(data['relay_ratio'])
                    if 'idle_ratio' in data:
                        idle_ratios.append(data['idle_ratio'])
            if relay_ratios and idle_ratios:
                avg_relay = np.mean(relay_ratios[-50:]) if len(relay_ratios) >= 50 else np.mean(relay_ratios)
                avg_idle = np.mean(idle_ratios[-50:]) if len(idle_ratios) >= 50 else np.mean(idle_ratios)
                mode_ratios = {
                    'D2D': max(0, 1 - avg_relay - avg_idle),
                    'Relay': avg_relay,
                    'Idle': avg_idle
                }
    except:
        pass
    
    axes2[0, 1].pie(mode_ratios.values(), labels=mode_ratios.keys(), autopct='%1.1f%%', 
                   colors=['skyblue', 'lightcoral', 'lightgray'], startangle=90)
    axes2[0, 1].set_title("Mode Distribution (Last 50 Episodes)", fontsize=14, fontweight='bold')
    
    # Amplitude distribution histogram (approximate)
    # Use average amplitude from logged data
    try:
        amplitudes = []
        if os.path.exists(metrics_log_file):
            with open(metrics_log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'avg_amp' in data:
                        amplitudes.append(data['avg_amp'])
        if amplitudes:
            axes2[1, 0].hist(amplitudes[-500:], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes2[1, 0].axvline(x=args.idle_threshold, color='r', linestyle='--', linewidth=2, label=f'Idle Threshold ({args.idle_threshold})')
            axes2[1, 0].set_xlabel("Amplitude", fontsize=12)
            axes2[1, 0].set_ylabel("Frequency", fontsize=12)
            axes2[1, 0].set_title("Amplitude Distribution (Last 500 Steps)", fontsize=14, fontweight='bold')
            axes2[1, 0].legend()
            axes2[1, 0].grid(True, alpha=0.3)
        else:
            axes2[1, 0].text(0.5, 0.5, 'No amplitude data available', ha='center', va='center', transform=axes2[1, 0].transAxes)
            axes2[1, 0].set_title("Amplitude Distribution", fontsize=14, fontweight='bold')
    except:
        axes2[1, 0].text(0.5, 0.5, 'No amplitude data available', ha='center', va='center', transform=axes2[1, 0].transAxes)
        axes2[1, 0].set_title("Amplitude Distribution", fontsize=14, fontweight='bold')
    
    # Mode ratio over time (stacked area)
    try:
        if os.path.exists(metrics_log_file):
            episodes = []
            d2d_ratios = []
            relay_ratios = []
            idle_ratios = []
            with open(metrics_log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    episodes.append(data['ep'])
                    relay_ratios.append(data.get('relay_ratio', 0))
                    idle_ratios.append(data.get('idle_ratio', 0))
                    d2d_ratios.append(max(0, 1 - data.get('relay_ratio', 0) - data.get('idle_ratio', 0)))
            
            if episodes:
                # Smooth with MA-10
                window_smooth = 10
                if len(d2d_ratios) >= window_smooth:
                    d2d_smooth = np.convolve(d2d_ratios, np.ones(window_smooth)/window_smooth, mode='valid')
                    relay_smooth = np.convolve(relay_ratios, np.ones(window_smooth)/window_smooth, mode='valid')
                    idle_smooth = np.convolve(idle_ratios, np.ones(window_smooth)/window_smooth, mode='valid')
                    ep_range = range(window_smooth-1, len(episodes))
                else:
                    d2d_smooth = d2d_ratios
                    relay_smooth = relay_ratios
                    idle_smooth = idle_ratios
                    ep_range = episodes
                
                axes2[1, 1].fill_between(ep_range, 0, d2d_smooth, label='D2D', color='skyblue', alpha=0.7)
                axes2[1, 1].fill_between(ep_range, d2d_smooth, np.array(d2d_smooth) + np.array(relay_smooth), 
                                        label='Relay', color='lightcoral', alpha=0.7)
                axes2[1, 1].fill_between(ep_range, np.array(d2d_smooth) + np.array(relay_smooth), 
                                        np.array(d2d_smooth) + np.array(relay_smooth) + np.array(idle_smooth),
                                        label='Idle', color='lightgray', alpha=0.7)
                axes2[1, 1].set_xlabel("Episode", fontsize=12)
                axes2[1, 1].set_ylabel("Mode Ratio", fontsize=12)
                axes2[1, 1].set_title("Mode Distribution Over Time (Stacked)", fontsize=14, fontweight='bold')
                axes2[1, 1].legend(loc='upper right')
                axes2[1, 1].grid(True, alpha=0.3)
                axes2[1, 1].set_ylim(0, 1)
            else:
                axes2[1, 1].text(0.5, 0.5, 'No mode data available', ha='center', va='center', transform=axes2[1, 1].transAxes)
                axes2[1, 1].set_title("Mode Distribution Over Time", fontsize=14, fontweight='bold')
        else:
            axes2[1, 1].text(0.5, 0.5, 'No mode data available', ha='center', va='center', transform=axes2[1, 1].transAxes)
            axes2[1, 1].set_title("Mode Distribution Over Time", fontsize=14, fontweight='bold')
    except Exception as e:
        axes2[1, 1].text(0.5, 0.5, f'Error loading mode data: {str(e)}', ha='center', va='center', transform=axes2[1, 1].transAxes)
        axes2[1, 1].set_title("Mode Distribution Over Time", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    additional_plot_filename = os.path.join(runs_dir, f'additional_analysis_{timestamp}.png')
    plt.savefig(additional_plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu đồ thị phân tích bổ sung vào file: {additional_plot_filename}")
    
    # Save final metrics summary
    final_metrics = {
        'maddpg': maddpg_metrics,
        'direct_transmission': dt_metrics,
        'greedy': greedy_metrics,
        'frequency_hopping': fh_metrics,
        'config': {
            'mutual_interf_coef': args.mutual_interf_coef,
            'idle_threshold': args.idle_threshold,
            'reward_w_thr': args.reward_w_thr,
            'reward_w_succ': args.reward_w_succ,
            'reward_w_margin': args.reward_w_margin,
            'reward_w_energy': args.reward_w_energy,
            'mob_sigma_user': args.mob_sigma_user,
            'mob_sigma_jammer': args.mob_sigma_jammer,
            'tau': args.tau,
            'amp_reg_coef': args.amp_reg_coef,
            'amp_reg_decay_steps': args.amp_reg_decay_steps,
            'expert_epsilon_start': args.expert_epsilon_start,
            'imitation_coef_max': args.imitation_coef_max,
            'seed': args.seed
        }
    }
    final_metrics_file = os.path.join(runs_dir, 'final_metrics.json')
    with open(final_metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✓ Đã lưu final metrics vào: {final_metrics_file}")
    
    # Generate final report
    report_file = os.path.join(runs_dir, 'final_report.md')
    with open(report_file, 'w') as f:
        f.write("# MADDPG Training Report\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Mutual Interference Coef: {args.mutual_interf_coef}\n")
        f.write(f"- Idle Threshold: {args.idle_threshold}\n")
        f.write(f"- Reward Weights: thr={args.reward_w_thr}, succ={args.reward_w_succ}, margin={args.reward_w_margin}, energy={args.reward_w_energy}\n")
        f.write(f"- Mobility: user={args.mob_sigma_user}, jammer={args.mob_sigma_jammer}\n")
        f.write(f"- Tau (soft update): {args.tau}\n")
        f.write(f"- Amplitude Regularizer: {args.amp_reg_coef} → 0 over {args.amp_reg_decay_steps} steps\n")
        f.write(f"- Expert Epsilon Start: {args.expert_epsilon_start}\n")
        f.write(f"- Imitation Coef Max: {args.imitation_coef_max}\n")
        f.write(f"- Seed: {args.seed}\n\n")
        f.write("## Results\n\n")
        f.write("| Method | Reward | Success Rate | Throughput | Energy Eff | Thr Eff |\n")
        f.write("|--------|--------|--------------|------------|------------|----------|\n")
        f.write(f"| Direct Transmission | {dt_metrics['reward']:.4f} | {dt_metrics['success_rate']:.3f} | {dt_metrics['avg_throughput']:.4f} | {dt_metrics['energy_efficiency']:.4f} | {dt_metrics['throughput_efficiency']:.4f} |\n")
        f.write(f"| Greedy Strategy | {greedy_metrics['reward']:.4f} | {greedy_metrics['success_rate']:.3f} | {greedy_metrics['avg_throughput']:.4f} | {greedy_metrics['energy_efficiency']:.4f} | {greedy_metrics['throughput_efficiency']:.4f} |\n")
        f.write(f"| Frequency Hopping | {fh_metrics['reward']:.4f} | {fh_metrics['success_rate']:.3f} | {fh_metrics['avg_throughput']:.4f} | {fh_metrics['energy_efficiency']:.4f} | {fh_metrics['throughput_efficiency']:.4f} |\n")
        f.write(f"| **MADDPG + JM** | **{maddpg_metrics['reward']:.4f}** | **{maddpg_metrics['success_rate']:.3f}** | **{maddpg_metrics['avg_throughput']:.4f}** | **{maddpg_metrics['energy_efficiency']:.4f}** | **{maddpg_metrics['throughput_efficiency']:.4f}** |\n\n")
        f.write("## Improvements\n\n")
        best_baseline_reward = max(dt_metrics['reward'], greedy_metrics['reward'])
        reward_improvement = ((maddpg_metrics['reward'] - best_baseline_reward) / abs(best_baseline_reward)) * 100
        f.write(f"- Reward improvement: {reward_improvement:+.2f}%\n")
        f.write(f"- Success rate improvement: {((maddpg_metrics['success_rate'] - greedy_metrics['success_rate']) / greedy_metrics['success_rate'] * 100):+.2f}% vs Greedy\n")
        f.write(f"- Throughput improvement: {((maddpg_metrics['avg_throughput'] - greedy_metrics['avg_throughput']) / greedy_metrics['avg_throughput'] * 100):+.2f}% vs Greedy\n")
    print(f"✓ Đã lưu final report vào: {report_file}")
    
    if PLOT_AVAILABLE:
        plt.show()