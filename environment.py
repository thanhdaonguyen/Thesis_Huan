# environment.py

import numpy as np

class CommEnvironment:
    def __init__(self, num_agents=5, area_size=500, jammer_power=1.0, noise_power=1e-4, path_loss_exponent=2.0,
                 mutual_interf_coef=0.06, idle_threshold=0.10,
                 reward_w_thr=4.0, reward_w_succ=0.8, reward_w_margin=0.6, reward_w_energy=0.05,
                 mob_sigma_user=2.5, mob_sigma_jammer=3.0):
        print("Initializing Communication Environment...")
        self.num_agents = num_agents
        self.area_size = area_size
        self.jammer_power = jammer_power  # P_J
        self.noise_power = noise_power    # sigma_R^2
        self.path_loss_exponent = path_loss_exponent # alpha

        # Mutual interference and idle configuration
        self.mutual_interf_coef = mutual_interf_coef  # Co-channel interference coefficient between SUs
        self.idle_threshold = idle_threshold  # Amplitude threshold for idle mode
        
        # Mobility parameters (reduced noise for stability)
        self.mob_sigma_user = mob_sigma_user  # Gaussian std for SU/DU movement
        self.mob_sigma_jammer = mob_sigma_jammer  # Gaussian std for jammer movement

        # Reward weights
        # [QRL-PATCH] reward weights tune
        self.w_thr = reward_w_thr      # Throughput weight
        self.w_succ = reward_w_succ    # Success bonus weight (default 1.2, was 0.8)
        self.w_margin = reward_w_margin # SINR margin bonus weight (default 0.8, was 0.6)
        self.w_energy = reward_w_energy # Energy penalty weight

        # Các thực thể: SU, DU, Jammer, Relay Base Station (rBS)
        self.sus = None
        self.dus = None
        self.jammer = None
        self.rbs = np.array([area_size / 2, area_size / 2]) # rBS đặt ở trung tâm

        # Các tham số của Jamming Modulation (JM)
        # [QRL-PATCH] soften power gain
        self.pga_gain = 30  # was 100 - Hệ số khuếch đại cực đại 'a' (biên độ tối đa)
        
        # [QRL-PATCH] State augmentation
        self.state_dim = 9 # Kích thước vector trạng thái cho mỗi agent (added prev_sinr)
        self.action_dim = 2 # Kích thước vector hành động cho mỗi agent
        
        # [QRL-PATCH] block fading for mobility
        self.block_fading_K = 20   # update positions every 20 steps
        self._mobility_tick = 0
        
        # Metrics tracking
        self.total_steps = 0
        self.successful_transmissions = 0
        self.total_energy = 0
        self.total_throughput = 0  # Track cumulative throughput
        # [QRL-PATCH] gentler threshold
        self.sinr_threshold = 0.10  # was 0.15 - Standard threshold
        
        # [QRL-PATCH] Track previous SINR for state augmentation
        self.prev_sinr = np.zeros(self.num_agents)

        self.reset()
        print("Environment Initialized.")

    def reset(self):
        """Khởi tạo lại vị trí của các thực thể cho mỗi episode mới."""
        self.sus = np.random.rand(self.num_agents, 2) * self.area_size
        self.dus = np.random.rand(self.num_agents, 2) * self.area_size
        self.jammer = np.random.rand(1, 2) * self.area_size
        
        # [QRL-PATCH] reset mobility tick
        self._mobility_tick = 0
        
        # Reset metrics
        self.total_steps = 0
        self.successful_transmissions = 0
        self.total_energy = 0
        self.total_throughput = 0
        
        # [QRL-PATCH] Reset previous SINR
        self.prev_sinr = np.zeros(self.num_agents)
        
        # Trả về trạng thái ban đầu của tất cả các agent
        return self._get_states()
    
    def get_metrics(self):
        """Return current episode metrics"""
        if self.total_steps == 0:
            return {
                'success_rate': 0.0,
                'energy_efficiency': 0.0,
                'avg_energy_per_step': 0.0,
                'avg_throughput': 0.0,
                'throughput_efficiency': 0.0
            }
        return {
            'success_rate': self.successful_transmissions / (self.total_steps * self.num_agents),
            'energy_efficiency': self.successful_transmissions / max(self.total_energy, 1e-9),
            'avg_energy_per_step': self.total_energy / self.total_steps,
            'avg_throughput': self.total_throughput / (self.total_steps * self.num_agents),
            'throughput_efficiency': self.total_throughput / max(self.total_energy, 1e-9)
        }

    def _get_distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2)**2))

    def _get_channel_gain(self, pos1, pos2):
        dist = self._get_distance(pos1, pos2)
        # Để tránh chia cho 0, thêm một epsilon nhỏ
        return (dist + 1e-6) ** -self.path_loss_exponent

    def _get_states(self):
        """Tạo vector trạng thái cho tất cả các agent."""
        states = []
        jnr = self.jammer_power / self.noise_power

        for i in range(self.num_agents):
            su_pos = self.sus[i]
            du_pos = self.dus[i]
            jammer_pos = self.jammer[0]

            # Các hệ số kênh liên quan đến agent i
            g_su_du = self._get_channel_gain(su_pos, du_pos)
            g_su_rbs = self._get_channel_gain(su_pos, self.rbs)
            g_rbs_du = self._get_channel_gain(self.rbs, du_pos)
            
            # Các hệ số kênh liên quan đến jammer
            g_jam_su = self._get_channel_gain(jammer_pos, su_pos)  # h1^2
            g_jam_du = self._get_channel_gain(jammer_pos, du_pos)  # h3^2
            g_jam_rbs = self._get_channel_gain(jammer_pos, self.rbs)
            
            # Vector trạng thái: JNR và các hệ số kênh quan trọng
            state = [
                np.log10(jnr + 1e-9),
                np.log10(g_su_du + 1e-9),
                np.log10(g_su_rbs + 1e-9),
                np.log10(g_rbs_du + 1e-9),
                np.log10(g_jam_su + 1e-9),
                np.log10(g_jam_du + 1e-9),
                np.log10(g_jam_rbs + 1e-9),
                self._get_distance(su_pos, du_pos) / self.area_size, # Khoảng cách chuẩn hóa
                self.prev_sinr[i] # [QRL-PATCH] Previous SINR (normalized/clipped implicitly by nature of SINR usually being small, but maybe log it?)
                # Let's keep it raw for now as it's usually < 10. Or maybe log10? 
                # SINR can be 0. Let's use log10(sinr + 1e-9) to be consistent with gains?
                # Actually, raw SINR is fine if clipped. Let's use the same clipping as in step: clip(sinr, 0, 10)
            ]
            states.append(np.array(state, dtype=np.float32))
        return states

    def step(self, actions):
        """
        Thực thi hành động của tất cả các agent và trả về trạng thái mới, phần thưởng.
        `actions` là một list, mỗi phần tử là hành động của một agent.
        Hành động của mỗi agent là một vector 2 chiều:
        - action[0]: Biên độ khuếch đại liên tục ∈ [-1, 1] → map to [0, 1]
        - action[1]: Lựa chọn chế độ giao tiếp (> 0 là Relay, <= 0 là D2D)
        """
        rewards = []
        info = {'sinr_values': [], 'modes': [], 'pga_choices': [], 'amplitudes': [], 'energy': []}
        
        # [QRL-PATCH] block-fading mobility
        if self._mobility_tick % self.block_fading_K == 0:
            # Di chuyển các thực thể một cách ngẫu nhiên (reduced noise for stability)
            self.sus += np.random.randn(self.num_agents, 2) * self.mob_sigma_user
            self.dus += np.random.randn(self.num_agents, 2) * self.mob_sigma_user
            self.jammer += np.random.randn(1, 2) * self.mob_sigma_jammer

            # Giữ các thực thể trong vùng mô phỏng
            self.sus = np.clip(self.sus, 0, self.area_size)
            self.dus = np.clip(self.dus, 0, self.area_size)
            self.jammer = np.clip(self.jammer, 0, self.area_size)
        self._mobility_tick += 1
        
        self.total_steps += 1

        # First pass: decode all actions and determine transmission status
        decoded_actions = []
        for i in range(self.num_agents):
            action = actions[i]
            # Map action[0] from [-1, 1] to [0, 1] (amplitude)
            a_raw = np.clip(float(action[0]), -1.0, 1.0)
            a = (a_raw + 1.0) / 2.0  # Now a ∈ [0, 1]
            
            # Determine if transmitting (idle if amplitude too low)
            tx = a >= self.idle_threshold
            
            # Calculate actual PGA gain
            pga_i = self.pga_gain * a if tx else 0.0
            
            # Mode selection
            mode = 'relay' if action[1] > 0 else 'd2d'
            
            decoded_actions.append({
                'amplitude': a,
                'tx': tx,
                'pga': pga_i,
                'mode': mode
            })

        # Second pass: compute SINR with mutual interference
        for i in range(self.num_agents):
            da = decoded_actions[i]
            su_pos = self.sus[i]
            du_pos = self.dus[i]
            jammer_pos = self.jammer[0]
            
            # Channel gains for agent i
            h1_sq = self._get_channel_gain(jammer_pos, su_pos)
            h3_sq = self._get_channel_gain(jammer_pos, du_pos)
            
            # Calculate mutual interference from other transmitting SUs
            inter_du = 0.0  # Interference at DU(i)
            inter_rbs = 0.0  # Interference at rBS
            
            for k in range(self.num_agents):
                if k != i and decoded_actions[k]['tx']:
                    pga_k = decoded_actions[k]['pga']
                    # Interference at DU(i) from SU(k)
                    g_k_to_du_i = self._get_channel_gain(self.sus[k], du_pos)
                    inter_du += self.mutual_interf_coef * g_k_to_du_i * (pga_k ** 2)
                    
                    # Interference at rBS from SU(k)
                    g_k_to_rbs = self._get_channel_gain(self.sus[k], self.rbs)
                    inter_rbs += self.mutual_interf_coef * g_k_to_rbs * (pga_k ** 2)
            
            # Calculate SINR based on mode
            sinr = 0.0
            pga_i = da['pga']
            
            if not da['tx']:
                # Idle mode: no transmission, SINR = 0
                sinr = 0.0
            elif da['mode'] == 'd2d':
                # D2D mode
                h2_sq = self._get_channel_gain(su_pos, du_pos)
                numerator = h1_sq * h2_sq * (pga_i ** 2) * self.jammer_power
                denominator = h3_sq * self.jammer_power + self.noise_power + inter_du
                sinr = numerator / (denominator + 1e-9)
                # [QRL-PATCH] clip SINR lower to prevent spikes
                sinr = float(np.clip(sinr, 1e-6, 10.0))
                
            elif da['mode'] == 'relay':
                # Relay mode: two-hop
                # Hop 1: SU -> rBS
                h_jam_rbs_sq = self._get_channel_gain(jammer_pos, self.rbs)
                h_su_rbs_sq = self._get_channel_gain(su_pos, self.rbs)
                num1 = h1_sq * h_su_rbs_sq * (pga_i ** 2) * self.jammer_power
                den1 = h_jam_rbs_sq * self.jammer_power + self.noise_power + inter_rbs
                sinr1 = num1 / (den1 + 1e-9)
                # [QRL-PATCH] clip SINR lower to prevent spikes
                sinr1 = float(np.clip(sinr1, 1e-6, 10.0))
                
                # Hop 2: rBS -> DU (assume rBS uses same PGA)
                h1_relay_sq = self._get_channel_gain(jammer_pos, self.rbs)
                h2_relay_sq = self._get_channel_gain(self.rbs, du_pos)
                h3_relay_sq = self._get_channel_gain(jammer_pos, du_pos)
                num2 = h1_relay_sq * h2_relay_sq * (pga_i ** 2) * self.jammer_power
                den2 = h3_relay_sq * self.jammer_power + self.noise_power + inter_du
                sinr2 = num2 / (den2 + 1e-9)
                # [QRL-PATCH] clip SINR lower to prevent spikes
                sinr2 = float(np.clip(sinr2, 1e-6, 10.0))
                
                sinr = min(sinr1, sinr2)
            
            # Track success
            is_successful = sinr > self.sinr_threshold
            if is_successful:
                self.successful_transmissions += 1
            
            # Calculate throughput (bits/s/Hz)
            throughput = np.log2(1 + sinr)
            self.total_throughput += throughput
            
            # Calculate energy cost
            if da['tx']:
                energy_cost = (da['amplitude'] ** 2) * (1.5 if da['mode'] == 'relay' else 1.0)
            else:
                energy_cost = 0.0
            self.total_energy += energy_cost
            
            # Calculate SINR margin (normalized excess above threshold)
            margin = max(0.0, (sinr - self.sinr_threshold) / (self.sinr_threshold + 1e-9))
            
            # [QRL-PATCH] idle bonus: reward not transmitting when channel is bad
            idle_bonus = 0.01 if (not da['tx'] and max(0.0, sinr) < self.sinr_threshold * 0.9) else 0.0
            
            # New reward function: R = w_thr*throughput + w_succ*success + w_margin*margin - w_energy*energy
            reward = (self.w_thr * throughput + 
                     self.w_succ * (1.0 if is_successful else 0.0) + 
                     self.w_margin * margin - 
                     self.w_energy * energy_cost +
                     idle_bonus)
            
            rewards.append(reward)
            
            # Track info for analysis (enhanced debug info)
            actual_mode = 'idle' if not da['tx'] else da['mode']
            info['sinr_values'].append(sinr)
            info['modes'].append(actual_mode)
            info['pga_choices'].append(pga_i)
            info['amplitudes'].append(da['amplitude'])
            info['amplitudes'].append(da['amplitude'])
            info['energy'].append(energy_cost)
            
            # [QRL-PATCH] Update prev_sinr for next step
            self.prev_sinr[i] = sinr
            
        next_states = self._get_states()
        dones = [False] * self.num_agents # Giả định episode không bao giờ kết thúc sớm

        return next_states, rewards, dones, info