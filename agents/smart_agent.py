import math
import numpy as np
import pooltool as pt
import copy
import random
from .agent import Agent

class SmartAgent(Agent):
    """
    SmartAgent (Optimized): 
    1. 严格限制解球（Kick Shot）的触发条件，仅在无直球时考虑，并增加难度惩罚。
    2. 改进安全球（Safety Shot）逻辑，确保力度足够吃库，避免简单犯规。
    3. MCTS 专注于评估高质量候选，减少对低概率方案的计算浪费。
    """
    
    def __init__(self):
        super().__init__()
        # MCTS 参数
        self.n_simulations = 40
        self.c_puct = 1.414
        self.max_candidates = 8
        
        # 物理模拟参数
        self.ball_radius = 0.028575
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        # 桌面边界
        self.table_w = 1.9812 
        self.table_l = 0.9906
        
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None:
            return self._random_action()
            
        if hasattr(table, 'w'): self.table_w = table.w
        if hasattr(table, 'l'): self.table_l = table.l

        # 1. 确定目标球
        active_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not active_targets:
            if "8" in balls and balls["8"].state.s != 4:
                active_targets = ["8"]
            else:
                return self._random_action()

        # 2. 生成候选动作
        candidates = self._generate_all_candidates(balls, active_targets, table)
        
        if not candidates:
            print("[SmartAgent] 无候选方案，尝试随机兜底")
            return self._random_safe_action(balls, table)
            
        # 3. 筛选候选
        candidates.sort(key=lambda x: x.get('difficulty', 999))
        candidates = candidates[:self.max_candidates]
        
        print(f"[SmartAgent] MCTS 搜索开始，候选数: {len(candidates)}")
        
        # 4. MCTS 搜索
        self._run_mcts(candidates, balls, table, active_targets)
        
        # 5. 鲁棒性检查 (Robustness Check)
        # 按 MCTS 分数排序
        candidates.sort(key=lambda x: x.get('score', -999), reverse=True)
        
        for cand in candidates[:3]: # 检查前3名
            if cand.get('score', -999) < -50: continue # 分数太低的不看
            
            pass_rate = self._check_robustness(cand, balls, table, active_targets)
            print(f"[SmartAgent] 方案 {cand['shot_type']} 目标:{cand['target_ball']} 鲁棒性:{pass_rate:.1f}")
            
            if pass_rate >= 0.6:
                return cand['action']
        
        print("[SmartAgent] 所有最佳方案鲁棒性不足，尝试高级兜底")
        return self._monte_carlo_fallback(balls, table, active_targets)

    def _check_robustness(self, candidate, balls, table, targets, n_checks=10):
        """对候选方案进行带噪声的多次验证"""
        success_count = 0
        for _ in range(n_checks):
            shot = self._simulate_action(balls, table, candidate['action'])
            if shot:
                score = self._analyze_shot_reward(shot, balls, targets)
                
                # 零容忍策略：只要出现一次直接判负（黑八提前入袋或母球黑八同进），立即判死刑
                if score <= -400: # -500 是判负分
                    print(f"[SmartAgent] 触发死亡红线！方案 {candidate['shot_type']} 导致直接判负")
                    return 0.0
                
                if score >= 0: # 非犯规
                    success_count += 1
        return success_count / n_checks

    def _monte_carlo_fallback(self, balls, table, targets):
        """蒙特卡洛随机兜底：生成多个随机动作，选最好的"""
        best_action = None
        best_score = -9999
        
        # 生成 20 个随机动作
        for _ in range(20):
            action = self._random_action()
            # 随机动作力度大一点，避免未碰库犯规
            action['V0'] = np.random.uniform(2.0, 5.0) 
            
            shot = self._simulate_action(balls, table, action)
            if shot:
                score = self._analyze_shot_reward(shot, balls, targets)
                # 稍微偏向于碰到任意球的
                if score > best_score:
                    best_score = score
                    best_action = action
        
        if best_action:
            print(f"[SmartAgent] 随机兜底成功找到方案 (得分: {best_score})")
            return best_action
        
        return self._random_safe_action(balls, table)

    # ================= MCTS 核心逻辑 =================
    
    def _run_mcts(self, candidates, balls, table, targets):
        n_candidates = len(candidates)
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        for i in range(self.n_simulations):
            # Selection
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb)
            
            # Simulation
            shot = self._simulate_action(balls, table, candidates[idx]['action'])
            
            # Evaluation
            if shot is None:
                score = -500.0
            else:
                score = self._analyze_shot_reward(shot, last_state_snapshot, targets)
                
                # 额外的走位奖励
                if score > 50:
                    cue_end = shot.balls['cue'].state.rvw[0]
                    dist = np.linalg.norm(cue_end[:2])
                    score += (1.0 - dist / (self.table_w/2)) * 30 
            
            # Normalization
            norm_score = np.clip((score - (-500)) / 700.0, 0.0, 1.0)
            
            # Backprop
            N[idx] += 1
            Q[idx] += norm_score
            
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        
        # 记录分数到所有候选对象中
        for i in range(n_candidates):
            candidates[i]['score'] = avg_rewards[i] * 700 - 500
        
        print(f"[SmartAgent] 最佳方案: {candidates[best_idx].get('shot_type', 'UNKNOWN')} "
              f"目标:{candidates[best_idx].get('target_ball')} "
              f"得分:{avg_rewards[best_idx]:.3f}")
              
        # return candidates[best_idx] # 不需要返回，因为已经记录了分数

    def _simulate_action(self, balls, table, action):
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            noisy_action = copy.deepcopy(action)
            noisy_action['V0'] = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 9.0)
            noisy_action['phi'] = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            cue.set_state(**noisy_action)
            shot = pt.System(table=table, balls=sim_balls, cue=cue)
            pt.simulate(shot, inplace=True)
            return shot
        except:
            return None

    # ================= 候选生成 (优化版) =================
    
    def _generate_all_candidates(self, balls, targets, table):
        candidates = []
        
        # 1. 直球方案
        direct_shots = self._generate_direct_shots(balls, targets, table)
        candidates.extend(direct_shots)
        
        # 策略调整：只要有合格的直球（难度<50），就完全不考虑解球
        # 避免为了万分之一的解球概率去浪费计算资源
        best_difficulty = min([x['difficulty'] for x in direct_shots]) if direct_shots else 999
        
        if not direct_shots or best_difficulty > 50:
            print("[SmartAgent] 直球困难/无直球，尝试生成解球...")
            kick_shots = self._generate_kick_shots(balls, targets, table)
            candidates.extend(kick_shots)
            
        # 2. 安全球方案 (增强版)
        # 无论有没有直球，都可以考虑安全球作为备选，特别是直球难度大时
        if len(candidates) < 5 or best_difficulty > 30:
            safety_shots = self._generate_safety_shots(balls, targets, table)
            candidates.extend(safety_shots)
            
        return candidates

    def _generate_direct_shots(self, balls, targets, table):
        candidates = []
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                to_pocket = pocket_pos - target_pos
                to_pocket[2] = 0
                dist_to_pocket = np.linalg.norm(to_pocket)
                if dist_to_pocket < 1e-6: continue
                
                unit_to_pocket = to_pocket / dist_to_pocket
                ghost_pos = target_pos - unit_to_pocket * (2 * R)
                
                aim_vector = ghost_pos - cue_pos
                aim_vector[2] = 0
                dist_cue_to_ghost = np.linalg.norm(aim_vector)
                
                if dist_cue_to_ghost < 1e-6: continue

                dot = np.dot(aim_vector / dist_cue_to_ghost, unit_to_pocket)
                cut_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                if cut_angle > 80: continue

                if self._is_path_blocked(cue_pos, ghost_pos, balls, ['cue', target_id]): continue
                if self._is_path_blocked(target_pos, pocket_pos, balls, ['cue', target_id]): continue

                phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0])) % 360
                V0 = np.clip(1.0 + (dist_cue_to_ghost + dist_to_pocket) * 1.8, 1.0, 5.0)
                
                for v_factor in [1.0, 1.1]:
                    candidates.append({
                        'shot_type': 'DIRECT',
                        'target_ball': target_id,
                        'pocket_id': pocket_id,
                        'action': {'V0': V0 * v_factor, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0},
                        'difficulty': cut_angle + dist_cue_to_ghost * 5
                    })
        return candidates

    def _generate_kick_shots(self, balls, targets, table):
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        rails = [
            {'name': 'right', 'normal': [-1, 0], 'x': self.table_w/2},
            {'name': 'left',  'normal': [1, 0],  'x': -self.table_w/2},
            {'name': 'top',   'normal': [0, -1], 'y': self.table_l/2},
            {'name': 'bottom','normal': [0, 1],  'y': -self.table_l/2},
        ]
        
        for target_id in targets:
            target_pos = balls[target_id].state.rvw[0]
            aim_point = target_pos
            
            for rail in rails:
                mirror = np.copy(aim_point)
                if 'x' in rail: mirror[0] = 2 * rail['x'] - aim_point[0]
                else: mirror[1] = 2 * rail['y'] - aim_point[1]
                
                aim_vec = mirror - cue_pos
                aim_vec[2] = 0
                dist = np.linalg.norm(aim_vec)
                
                t = 0.5
                if 'x' in rail: t = (rail['x'] - cue_pos[0]) / (mirror[0] - cue_pos[0])
                else: t = (rail['y'] - cue_pos[1]) / (mirror[1] - cue_pos[1])
                
                if not (0 <= t <= 1): continue
                hit_point = cue_pos + t * (mirror - cue_pos)
                
                if self._is_path_blocked(cue_pos, hit_point, balls, ['cue']): continue
                if self._is_path_blocked(hit_point, aim_point, balls, ['cue', target_id]): continue
                
                phi = np.degrees(np.arctan2(aim_vec[1], aim_vec[0])) % 360
                V0 = np.clip(2.0 + dist * 2.5, 2.5, 7.5) # 增加力度以补偿反弹损失
                
                candidates.append({
                    'shot_type': 'KICK',
                    'target_ball': target_id,
                    'pocket_id': 'kick',
                    'action': {'V0': V0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0},
                    'difficulty': 100 + dist * 20 # 大幅增加基础难度分，使其排序靠后
                })
        return candidates

    def _generate_safety_shots(self, balls, targets, table):
        """生成安全球 (改进版：确保力度足够碰库)"""
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for tid in targets:
            target_pos = balls[tid].state.rvw[0]
            aim_vec = target_pos - cue_pos
            if self._is_path_blocked(cue_pos, target_pos, balls, ['cue', tid]): continue
            
            phi = np.degrees(np.arctan2(aim_vec[1], aim_vec[0])) % 360
            
            # 改进：力度不能太小，至少要让目标球碰库
            # 简单估算：只要比之前大一点
            candidates.append({
                'shot_type': 'SAFETY',
                'target_ball': tid,
                'pocket_id': 'safe',
                'action': {'V0': 2.5, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}, # 增加到 2.5
                'difficulty': 40
            })
        return candidates

    def _is_path_blocked(self, start, end, balls, exclude_ids):
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6: return False
        path_dir = path_vec / path_len
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4: continue
            ball_pos = ball.state.rvw[0]
            to_ball = ball_pos - start
            proj = np.dot(to_ball, path_dir)
            if 0 < proj < path_len:
                perp = np.linalg.norm(to_ball - proj * path_dir)
                if perp < 2 * ball.params.R * 0.95: return True
        return False
        
    def _analyze_shot_reward(self, shot, last_state, player_targets):
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        foul_first_hit = False
        first_contact = None
        valid_balls = set(last_state.keys()) - {'cue'}
        
        # 检查是否碰库
        any_cushion_hit = False
        
        for e in shot.events:
            ids = list(e.ids) if hasattr(e, 'ids') else []
            event_type = str(e.event_type).lower()
            
            if 'cushion' in event_type:
                any_cushion_hit = True
                
            if 'cue' in ids and 'cushion' not in event_type and 'pocket' not in event_type:
                other = [x for x in ids if x != 'cue']
                if other and other[0] in valid_balls:
                    if first_contact is None: first_contact = other[0]
        
        if first_contact is None or first_contact not in player_targets:
            foul_first_hit = True

        # 计分
        if cue_pocketed and eight_pocketed: return -500
        if cue_pocketed: return -100
        if eight_pocketed:
            if len(player_targets) == 1 and player_targets[0] == "8": return 200
            else: return -500

        score = 0
        if foul_first_hit: score -= 50
        
        # 检查无进球且未碰库犯规
        if not new_pocketed and not any_cushion_hit and not foul_first_hit:
            score -= 30 # 未碰库犯规惩罚
            
        score += len(own_pocketed) * 100
        score -= len(enemy_pocketed) * 30
        
        if score == 0 and not foul_first_hit and any_cushion_hit: 
            score = 10 # 合法防守
            
        return score

    def _random_safe_action(self, balls, table):
        # 暴力防守：朝球堆中心打一杆，或者随机打
        # 这里简单点，还是随机，但稍微增加力度以确保碰库
        action = self._random_action()
        action['V0'] = max(action['V0'], 2.0)
        return action
