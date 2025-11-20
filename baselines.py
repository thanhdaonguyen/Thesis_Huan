# baselines.py

import numpy as np

class DirectTransmission:
    """
    Chiến lược cơ sở: Luôn cố gắng truyền tin một cách ngây thơ.
    Nó không quan tâm đến trạng thái của môi trường.
    Hành động: Luôn gửi bit '1' (khuếch đại) và dùng chế độ D2D.
    """
    def __init__(self, num_agents, action_dim):
        self.num_agents = num_agents
        self.action_dim = action_dim
        # Hành động cố định: [pga_choice=1, mode_choice=-1 (D2D)]
        self.fixed_action = np.array([1.0, -1.0])

    def select_actions(self, states):
        # Trả về cùng một hành động cho tất cả các agent, bỏ qua trạng thái
        return [self.fixed_action for _ in range(self.num_agents)]

class GreedyStrategy:
    """
    Chiến lược tham lam: Tại mỗi bước, chọn hành động (chế độ D2D hoặc Relay)
    tối đa hóa SINR tức thời.
    """
    def __init__(self, num_agents, action_dim, env_params):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.pga_gain = env_params['pga_gain']
        self.jammer_power = env_params['jammer_power']
        self.noise_power = env_params['noise_power']

    def select_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            state = states[i]
            
            # Khôi phục các giá trị hệ số kênh từ trạng thái (đã được log10)
            g_su_du = 10**state[1]
            g_su_rbs = 10**state[2]
            g_rbs_du = 10**state[3]
            g_jam_su = 10**state[4]
            g_jam_du = 10**state[5]
            g_jam_rbs = 10**state[6]

            # --- Tính toán SINR cho chế độ D2D ---
            # Greedy luôn chọn gửi bit '1' (khuếch đại)
            pga_choice = self.pga_gain
            num_d2d = g_jam_su * g_su_du * (pga_choice**2) * self.jammer_power
            den_d2d = g_jam_du * self.jammer_power + self.noise_power
            sinr_d2d = num_d2d / (den_d2d + 1e-9)

            # --- Tính toán SINR cho chế độ Relay ---
            # Chặng 1: SU -> rBS
            num1_relay = g_jam_su * g_su_rbs * (pga_choice**2) * self.jammer_power
            den1_relay = g_jam_rbs * self.jammer_power + self.noise_power
            sinr1_relay = num1_relay / (den1_relay + 1e-9)
            
            # Chặng 2: rBS -> DU
            num2_relay = g_jam_rbs * g_rbs_du * (pga_choice**2) * self.jammer_power
            den2_relay = g_jam_du * self.jammer_power + self.noise_power
            sinr2_relay = num2_relay / (den2_relay + 1e-9)
            
            sinr_relay = min(sinr1_relay, sinr2_relay)

            # --- So sánh và đưa ra quyết định ---
            if sinr_d2d >= sinr_relay:
                # Chọn D2D
                action = np.array([1.0, -1.0]) 
            else:
                # Chọn Relay
                action = np.array([1.0, 1.0])
            
            actions.append(action)
        return actions

class FrequencyHopping:
    """
    Chiến lược kinh điển: Nhảy tần để tránh nhiễu.
    Để so sánh công bằng, tính reward theo cùng công thức như env (bao gồm energy penalty).
    """
    def __init__(self, num_agents, env_params, reward_w_thr=4.0, reward_w_succ=0.8, reward_w_margin=0.6, reward_w_energy=0.05, sinr_threshold=0.15, fh_fair=True, env=None):
        self.num_agents = num_agents
        # [QRL-PATCH] control fair/oracle FH
        self.fh_fair = fh_fair  # default True (settable from main)
        self.env = env  # [QRL-PATCH] need env for fair evaluation
        
        # Giả định có 40 kênh tần số, và jammer chiếm 10 kênh trong đó
        self.num_channels = 40
        self.num_jammed_channels = 10
        self.p_collision = self.num_jammed_channels / self.num_channels

        # Các tham số cho truyền thông bình thường (khi không bị nhiễu)
        self.tx_power = 0.1 # Công suất phát thông thường (Watt)
        self.noise_power = env_params['noise_power']
        
        # Reward weights (same as env for fair comparison)
        self.w_thr = reward_w_thr
        self.w_succ = reward_w_succ
        self.w_margin = reward_w_margin
        self.w_energy = reward_w_energy
        self.sinr_threshold = sinr_threshold
        
        # Estimate average amplitude for FH (assume ~0.5 on average when transmitting)
        self.avg_amplitude = 0.5

    def select_actions(self, states):
        """
        [QRL-PATCH] FH action selection: choose actions based on hop success/failure.
        If fh_fair=True, returns actions to be evaluated by env.step().
        """
        actions = []
        for i in range(self.num_agents):
            # Randomly decide if hop succeeds (not jammed)
            hop_succeeds = np.random.rand() > self.p_collision
            
            if hop_succeeds:
                # [QRL-PATCH] hop success: transmit with moderate amplitude, D2D mode
                # Map amplitude 0.5 to action space: 0.5 -> action[0] = 0.0 (middle of [-1,1])
                action = np.array([0.0, -1.0])  # amplitude=0.5 maps to 0.0, D2D mode
            else:
                # [QRL-PATCH] hop fail: don't transmit (idle)
                # Use amplitude < idle_threshold: map to action[0] < -0.64 (maps to ~0.18)
                # Actually, simpler: use -1.0 which maps to 0.0 amplitude (idle)
                action = np.array([-1.0, -1.0])  # idle (amplitude=0), mode doesn't matter
            
            actions.append(action)
        return actions
    
    def get_rewards(self, states):
        """
        [QRL-PATCH] Legacy oracle reward computation (used when fh_fair=False).
        Tính toán phần thưởng kỳ vọng cho chiến lược FH.
        Sử dụng cùng công thức reward như env để so sánh công bằng.
        """
        rewards = []
        for i in range(self.num_agents):
            state = states[i]
            # Lấy hệ số kênh SU-DU
            g_su_du = 10**state[1]

            # Khi hop thành công (không bị nhiễu)
            sinr_success = (self.tx_power * g_su_du) / (self.noise_power + 1e-9)
            sinr_success = float(np.clip(sinr_success, 1e-6, 10.0))  # [QRL-PATCH] clip like env (was 50.0)
            throughput_success = np.log2(1 + sinr_success)
            is_successful_success = sinr_success > self.sinr_threshold
            margin_success = max(0.0, (sinr_success - self.sinr_threshold) / (self.sinr_threshold + 1e-9))
            energy_cost = (self.avg_amplitude ** 2) * 1.0  # D2D mode assumed
            
            reward_success = (self.w_thr * throughput_success + 
                            self.w_succ * (1.0 if is_successful_success else 0.0) + 
                            self.w_margin * margin_success - 
                            self.w_energy * energy_cost)

            # Khi hop thất bại (rơi vào kênh nhiễu)
            sinr_fail = 0.0
            throughput_fail = np.log2(1 + sinr_fail)
            is_successful_fail = False
            margin_fail = 0.0
            # Still pay energy cost even if transmission fails
            reward_fail = (self.w_thr * throughput_fail + 
                          self.w_succ * (1.0 if is_successful_fail else 0.0) + 
                          self.w_margin * margin_fail - 
                          self.w_energy * energy_cost)

            # Phần thưởng kỳ vọng
            expected_reward = (1 - self.p_collision) * reward_success + self.p_collision * reward_fail
            rewards.append(expected_reward)
        
        return rewards