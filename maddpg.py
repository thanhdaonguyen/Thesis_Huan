# maddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# SIMPLIFIED POWERFUL Actor Network 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        # Simpler but effective architecture
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        
        # Simple but effective normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        # ReLU works well for this simpler network
        self.activation = nn.ReLU()
        # Dropout removed for cleaner training
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state):
        x = self.activation(self.ln1(self.fc1(state)))
        
        x = self.activation(self.ln2(self.fc2(x)))
        
        x = self.activation(self.ln3(self.fc3(x)))
        
        # Output with tanh activation for bounded actions
        action = torch.tanh(self.fc4(x))
        return action

# SIMPLIFIED POWERFUL Critic Network
class Critic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Critic processes all agents' states and actions
        input_dim = (state_dim + action_dim) * num_agents
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simpler but effective architecture
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
        # Simple normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(64)
        
        # ReLU activation
        self.activation = nn.ReLU()
        # Dropout removed for cleaner training
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, states, actions):
        # Concatenate all states and actions
        x = torch.cat(states + actions, dim=1)
        
        x = self.activation(self.ln1(self.fc1(x)))
        
        x = self.activation(self.ln2(self.fc2(x)))
        
        x = self.activation(self.ln3(self.fc3(x)))
        
        x = self.activation(self.ln4(self.fc4(x)))
        
        q_value = self.fc5(x)
        return q_value

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, agent_id, num_agents, lr_actor=1e-4, lr_critic=2e-4, gamma=0.98, tau=0.005, device=None, env_params=None,
                 epsilon_start=0.85, epsilon_end=0.05, epsilon_decay=0.9985,
                 noise_scale_start=0.15, noise_scale_end=0.05,
                 expert_epsilon_start=0.15, expert_decay=0.95, imitation_coef_max=0.10, imitation_warmup_steps=20000,
                 weight_decay=1e-5, amp_reg_coef=0.01, amp_reg_decay_steps=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device("cpu")
        
        # Gradient clipping (tighter for stability)
        self.max_grad_norm_actor = 1.0
        self.max_grad_norm_critic = 2.0
        
        # Improved exploration strategy (reduced wildness)
        # [QRL-PATCH] exploration schedule
        self.epsilon_start = epsilon_start if epsilon_start != 0.85 else 0.60
        self.epsilon_end = epsilon_end
        self.epsilon_min = 0.05  # [QRL-PATCH] min epsilon
        self.epsilon_decay = epsilon_decay if epsilon_decay != 0.9985 else 0.995
        self.epsilon = self.epsilon_start
        
        # Gaussian noise for exploration (linear decay)
        # [QRL-PATCH] quieter noise
        self.noise_scale_start = noise_scale_start if noise_scale_start != 0.15 else 0.08
        self.noise_scale_end = noise_scale_end
        self.noise_scale_min = 0.01  # [QRL-PATCH] min noise scale
        self.noise_decay = 0.995  # [QRL-PATCH] noise decay per episode
        self.noise_scale = self.noise_scale_start
        
        # Expert guidance (Greedy strategy for imitation) - further reduced
        self.env_params = env_params
        self.use_expert = True
        # [QRL-PATCH] expert gating exploration
        self.expert_epsilon = expert_epsilon_start if expert_epsilon_start != 0.15 else 0.10
        self.expert_epsilon_min = 0.02  # [QRL-PATCH] min expert epsilon
        self.expert_decay = expert_decay if expert_decay != 0.95 else 0.990
        self.imitation_coef_max = imitation_coef_max
        self.imitation_warmup_steps = imitation_warmup_steps
        
        # Amplitude regularizer with decay
        self.amp_reg_coef = amp_reg_coef
        self.amp_reg_decay_steps = amp_reg_decay_steps
        
        # Training tracking
        self.training_step = 0

        # Tạo mạng Actor và Critic, cùng với các mạng target của chúng
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(num_agents, state_dim, action_dim).to(self.device)
        self.target_critic = Critic(num_agents, state_dim, action_dim).to(self.device)

        # Sao chép trọng số ban đầu
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers with adjusted learning rates and weight decay
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    def get_expert_action(self, state):
        """Get greedy expert action for imitation learning with gating"""
        if self.env_params is None:
            return None
            
        # Decode state (same as in GreedyStrategy)
        g_su_du = 10**state[1]
        g_su_rbs = 10**state[2]
        g_rbs_du = 10**state[3]
        g_jam_su = 10**state[4]
        g_jam_du = 10**state[5]
        g_jam_rbs = 10**state[6]
        
        pga_choice = self.env_params['pga_gain']
        
        # Calculate SINR for D2D
        num_d2d = g_jam_su * g_su_du * (pga_choice**2) * self.env_params['jammer_power']
        den_d2d = g_jam_du * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr_d2d = num_d2d / (den_d2d + 1e-9)
        
        # Calculate SINR for Relay
        num1_relay = g_jam_su * g_su_rbs * (pga_choice**2) * self.env_params['jammer_power']
        den1_relay = g_jam_rbs * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr1_relay = num1_relay / (den1_relay + 1e-9)
        
        num2_relay = g_jam_rbs * g_rbs_du * (pga_choice**2) * self.env_params['jammer_power']
        den2_relay = g_jam_du * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr2_relay = num2_relay / (den2_relay + 1e-9)
        
        sinr_relay = min(sinr1_relay, sinr2_relay)
        
        # EXPERT GATING: Don't transmit when SINR too low
        thr = self.env_params.get("sinr_threshold", 0.15)
        if max(sinr_d2d, sinr_relay) < 0.9 * thr:
            # Return idle action: amplitude=-1 (maps to 0), mode doesn't matter
            return np.array([-1.0, -1.0], dtype=np.float32)
        
        # Choose best mode when SINR is acceptable
        if sinr_d2d >= sinr_relay:
            # D2D mode: action[0]=1 (send bit 1), action[1]=-1 (D2D)
            expert_action = np.array([1.0, -1.0], dtype=np.float32)
        else:
            # Relay mode: action[0]=1, action[1]=1 (Relay)
            expert_action = np.array([1.0, 1.0], dtype=np.float32)
            
        return expert_action
    
    def select_action(self, state, add_noise=True):
        state_np = state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Early training: use expert with probability (reduced)
        if add_noise and self.use_expert and self.env_params is not None and np.random.rand() < self.expert_epsilon:
            action = self.get_expert_action(state_np)
            if action is not None:
                # Add small noise to expert action for exploration
                action += np.random.normal(0, 0.1, self.action_dim)
                self.training_step += 1
                return np.clip(action, -1, 1)
        
        # Normal epsilon-greedy exploration (NO BIAS - allow idle and all modes)
        if add_noise and np.random.rand() < self.epsilon:
            action = np.random.uniform(-1, 1, self.action_dim)
            # No bias during warmup - let agent explore idle and all modes uniformly
            if self.training_step < self.imitation_warmup_steps:
                action[0] = np.random.uniform(-1.0, 1.0)  # Allow idle (a<0) and transmit (a>0)
                action[1] = np.random.uniform(-1.0, 1.0)  # Allow both D2D and Relay
        else:
            # Use actor network
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).squeeze(0).cpu().numpy()
            self.actor.train()
            
            if add_noise:
                # Gaussian noise (simpler than OU)
                noise = np.random.normal(0, self.noise_scale, self.action_dim)
                action += noise
        
        self.training_step += 1
        return np.clip(action, -1, 1)
    
    def reset_noise(self):
        """Reset noise and decay exploration parameters for new episode"""
        # [QRL-PATCH] decay epsilon each episode with min clamp
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # [QRL-PATCH] decay expert epsilon with min clamp
        self.expert_epsilon = max(self.expert_epsilon_min, self.expert_epsilon * self.expert_decay)
        # [QRL-PATCH] decay noise scale per episode
        self.noise_scale = max(self.noise_scale_min, self.noise_scale * self.noise_decay)
        
        # Disable expert after warmup
        if self.training_step >= self.imitation_warmup_steps:
            self.use_expert = False

    def amp_reg_weight(self):
        """Cosine decay schedule for amplitude regularizer: starts at amp_reg_coef, decays to 0"""
        import math
        w0 = self.amp_reg_coef
        T = self.amp_reg_decay_steps
        t = min(self.training_step, T)
        return w0 * 0.5 * (1.0 + math.cos(math.pi * t / T))  # Cosine decay to 0

    def update_targets(self):
        # Soft update cho các mạng target
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))