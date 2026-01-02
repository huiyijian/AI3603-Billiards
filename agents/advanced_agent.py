import math
import numpy as np
import pooltool as pt
import copy
import random
from .agent import Agent

class AdvancedAgent(Agent):
    """
    AdvancedAgent: 
    1. 继承 NewAgent 的几何规划优势
    2. 增加库边反弹（Kick Shot）逻辑，解决被斯诺克的问题
    3. 引入更完善的评分机制（参考 BasicAgentPro）
    4. 增加母球走位（Positioning）的简单评估
    """
    
    def __init__(self):
        super().__init__()
        self.num_simulation_samples = 15  # 增加采样数以提高稳定性
        self.max_candidates = 12         # 增加候选数以容纳反弹球方案
        self.rail_margin = 0.05          # 库边反弹计算的边界余量
        
        # 桌面边界 (标准 PoolTable 尺寸，近似值，实际应从 table 对象获取)
        # 这里先给个大概范围用于快速过滤，具体碰撞检测交给 pooltool
        self.table_w = 1.9812 # 宽
        self.table_l = 0.9906 # 高
        
    def decision(self, balls=None, my_targets=None, table=None):
        """核心决策函数"""
        if balls is None or table is None:
            return self._random_action()
            
        # 1. 确定目标球列表
        active_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not active_targets:
            if "8" in balls and balls["8"].state.s != 4:
                active_targets = ["8"]
            else:
                return self._random_action()

        # 更新桌面边界信息 (第一次调用时获取)
        if hasattr(table, 'w') and hasattr(table, 'l'):
            self.table_w = table.w
            self.table_l = table.l

        # 2. 生成候选击球方案 (直球 + 反弹球)
        candidates = self._generate_candidates(balls, active_targets, table)
        
        # 如果没有候选方案，尝试更激进的解球（Kick Shot 搜索）
        if not candidates:
            print("[AdvancedAgent] 无直球方案，尝试生成解球方案...")
            candidates = self._generate_kick_candidates(balls, active_targets, table)
        
        if not candidates:
            print("[AdvancedAgent] 无几何可行方案(含解球)，尝试随机击球")
            # 尝试安全球
            safety = self._find_safety_shot(balls, active_targets, table)
            return safety if safety else self._random_safe_action(balls, table)

        # 3. 模拟评估
        best_action = None
        best_score = -float('inf')
        
        # 排序并截断
        candidates.sort(key=lambda x: x['difficulty'])
        top_candidates = candidates[:self.max_candidates]

        print(f"[AdvancedAgent] 评估 {len(top_candidates)} 个方案 (target={active_targets})")

        is_clearing_black_eight = (len(active_targets) == 1 and active_targets[0] == "8")
        
        # 缓存初始状态用于评分
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        for cand in top_candidates:
            # 力度优化搜索
            base_V0 = cand['action']['V0']
            best_v_score = -float('inf')
            best_v = base_V0
            
            # 针对反弹球，力度可能需要更大，搜索范围稍微放宽
            v_factors = [0.9, 1.0, 1.1, 1.2] if cand.get('is_kick') else [0.9, 1.0, 1.1]
            
            for v_factor in v_factors:
                test_cand = copy.deepcopy(cand)
                test_cand['action']['V0'] = np.clip(base_V0 * v_factor, 0.5, 9.0)
                
                score = self._evaluate_candidate(
                    test_cand, balls, table, my_targets, 
                    is_clearing_black_eight, last_state_snapshot
                )
                
                if score > best_v_score:
                    best_v_score = score
                    best_v = test_cand['action']['V0']
            
            cand['action']['V0'] = best_v
            final_score = best_v_score

            # 打印调试信息
            shot_type = "KICK" if cand.get('is_kick') else "DIRECT"
            print(f"  - [{shot_type}] 目标:{cand['target_ball']} 袋口:{cand['pocket_id']} "
                  f"V0:{best_v:.1f} 分数:{final_score:.1f}")

            if final_score > best_score:
                best_score = final_score
                best_action = cand['action']

        # 决策逻辑
        if best_action and best_score > -50:
            return best_action
            
        print("[AdvancedAgent] 最佳方案分数过低，尝试安全球")
        safety_action = self._find_safety_shot(balls, active_targets, table)
        if safety_action:
            return safety_action
            
        # 如果有进攻方案但分数低（可能风险大），在没有安全球时还是得打
        if best_action:
            print("[AdvancedAgent] 无安全球，强制采用进攻方案")
            return best_action

        return self._random_safe_action(balls, table)

    # ================= 候选生成逻辑 =================
    
    def _generate_candidates(self, balls, targets, table):
        """生成直球候选"""
        candidates = []
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # Ghost Ball 计算
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

                # 切角检查
                dot_prod = np.dot(aim_vector / dist_cue_to_ghost, unit_to_pocket)
                cut_angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
                if cut_angle_deg > 80: continue

                # 路径障碍检查
                if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude_ids=['cue', target_id]):
                    continue
                if self._is_path_blocked(target_pos, pocket_pos, balls, exclude_ids=['cue', target_id]):
                    continue

                # 计算 phi
                phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0])) % 360
                
                # 估算力度
                total_dist = dist_cue_to_ghost + dist_to_pocket
                V0 = np.clip(1.0 + total_dist * 1.8, 1.0, 5.0)

                candidates.append({
                    'target_ball': target_id,
                    'pocket_id': pocket_id,
                    'action': {'V0': V0, 'phi': phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0},
                    'cut_angle': cut_angle_deg,
                    'difficulty': cut_angle_deg + dist_cue_to_ghost * 5,
                    'is_kick': False
                })
        return candidates

    def _generate_kick_candidates(self, balls, targets, table):
        """
        生成一库解球方案 (Kick Shot)
        原理：镜像法。计算幽灵球关于库边的镜像点。
        """
        candidates = []
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R
        
        # 定义 4 个库边的参数 (假设矩形桌面中心在 0,0)
        # pooltool 的 table.w 是宽(x轴方向?), table.l 是长(y轴方向?)
        # 需根据实际坐标系确认。通常 w 是短边，l 是长边。
        # 这里使用一种通用的镜像搜索方法
        
        rails = [
            {'name': 'right', 'normal': [-1, 0], 'x': self.table_w/2},
            {'name': 'left',  'normal': [1, 0],  'x': -self.table_w/2},
            {'name': 'top',   'normal': [0, -1], 'y': self.table_l/2},
            {'name': 'bottom','normal': [0, 1],  'y': -self.table_l/2},
        ]

        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 我们只需要碰到目标球即可（未必进球），因为是解球
            # 但为了尽可能进球，还是计算 Ghost Ball
            # 由于解球很难控制进袋，这里简化为：瞄准目标球中心（厚打）
            # 或者遍历几个袋口计算 Ghost Ball
            
            # 策略：尝试瞄准目标球中心（保本），以及尝试瞄准 Ghost Ball（进阶）
            # 这里先尝试瞄准目标球本身（至少碰到球不犯规）
            aim_point = target_pos 
            
            for rail in rails:
                # 1. 计算 aim_point 关于 rail 的镜像点
                mirror_point = np.copy(aim_point)
                if 'x' in rail: # 左右库
                    mirror_point[0] = 2 * rail['x'] - aim_point[0]
                else: # 上下库
                    mirror_point[1] = 2 * rail['y'] - aim_point[1]
                
                # 2. 母球瞄准镜像点
                aim_vector = mirror_point - cue_pos
                aim_vector[2] = 0
                dist_total = np.linalg.norm(aim_vector)
                
                if dist_total < 1e-6: continue
                
                # 3. 计算库边撞击点 (Intersection)
                # 简单的几何比例： Cue -> HitPoint -> AimPoint 是共线的（在镜像空间）
                # HitPoint 在库边上
                # 对于垂直/水平库边，直接计算交点
                hit_point = np.zeros(3)
                if 'x' in rail:
                    t = (rail['x'] - cue_pos[0]) / (mirror_point[0] - cue_pos[0])
                    if not (0 <= t <= 1): continue # 反向了
                    hit_point = cue_pos + t * (mirror_point - cue_pos)
                    # 检查 hit_point 是否在桌面上 (y 轴范围内)
                    if abs(hit_point[1]) > self.table_l/2: continue
                else:
                    t = (rail['y'] - cue_pos[1]) / (mirror_point[1] - cue_pos[1])
                    if not (0 <= t <= 1): continue
                    hit_point = cue_pos + t * (mirror_point - cue_pos)
                    # 检查 hit_point 是否在桌面上 (x 轴范围内)
                    if abs(hit_point[0]) > self.table_w/2: continue

                # 4. 路径检查
                # 第一段：Cue -> HitPoint
                if self._is_path_blocked(cue_pos, hit_point, balls, exclude_ids=['cue']):
                    continue
                # 第二段：HitPoint -> Target
                if self._is_path_blocked(hit_point, aim_point, balls, exclude_ids=['cue', target_id]):
                    continue
                
                # 5. 生成动作
                phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0])) % 360
                
                # 力度需要稍大，因为碰库有能量损失
                V0 = np.clip(1.5 + dist_total * 2.0, 2.0, 6.0)
                
                candidates.append({
                    'target_ball': target_id,
                    'pocket_id': 'kick_shot', # 虚拟袋口
                    'action': {'V0': V0, 'phi': phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0},
                    'cut_angle': 0, # 解球不论切角
                    'difficulty': 50 + dist_total * 10, # 基础难度设高点
                    'is_kick': True
                })
                
        return candidates

    def _is_path_blocked(self, start, end, balls, exclude_ids):
        """路径检测（沿用 NewAgent 逻辑）"""
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6: return False
        path_dir = path_vec / path_len
        
        for bid, ball in balls.items():
            if bid in exclude_ids: continue
            if ball.state.s == 4: continue
            
            ball_pos = ball.state.rvw[0]
            to_ball = ball_pos - start
            proj_len = np.dot(to_ball, path_dir)
            
            if 0 < proj_len < path_len:
                perp_dist = np.linalg.norm(to_ball - proj_len * path_dir)
                if perp_dist < 2 * ball.params.R * 0.92: # 稍微放宽判定
                    return True
        return False

    # ================= 评估与评分 =================
    
    def _evaluate_candidate(self, candidate, balls, table, my_targets, is_clearing, last_state_snapshot):
        """物理模拟评分"""
        total_score = 0
        success_count = 0
        
        for _ in range(self.num_simulation_samples):
            noisy_action = copy.deepcopy(candidate['action'])
            # 噪声注入
            noisy_action['V0'] += np.random.normal(0, 0.1)
            noisy_action['phi'] += np.random.normal(0, 0.15) # 角度噪声稍大
            noisy_action['phi'] %= 360
            
            # 模拟
            curr_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            cue.set_state(**noisy_action)
            shot = pt.System(table=table, balls=curr_balls, cue=cue)
            
            try:
                pt.simulate(shot, inplace=True)
            except:
                total_score -= 100
                continue
                
            # 评分
            score = self._analyze_shot_reward(shot, last_state_snapshot, my_targets)
            
            # 走位奖励 (Positioning Bonus)
            # 如果进球了，检查母球位置是否好
            if score > 50: # 判定为进球
                success_count += 1
                cue_end_pos = shot.balls['cue'].state.rvw[0]
                # 简单策略：母球尽量靠近中心，避免贴库
                dist_to_center = np.linalg.norm(cue_end_pos[:2])
                score += (1.0 - dist_to_center / (self.table_w/2)) * 20 # 最高加20分
            
            total_score += score
            
        avg_score = total_score / self.num_simulation_samples
        # 成功率加成
        success_rate = success_count / self.num_simulation_samples
        avg_score += success_rate * 50
        
        return avg_score

    def _analyze_shot_reward(self, shot, last_state, player_targets):
        """
        详细的规则评分（移植自 BasicAgentPro 并简化）
        """
        # 1. 进球分析
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        # 2. 犯规判定
        foul_first_hit = False
        first_contact = None
        
        # 寻找首个碰撞球
        valid_balls = set(last_state.keys()) - {'cue'}
        for e in shot.events:
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cue' in ids and 'cushion' not in str(e.event_type).lower() and 'pocket' not in str(e.event_type).lower():
                other = [x for x in ids if x != 'cue']
                if other and other[0] in valid_balls:
                    first_contact = other[0]
                    break
        
        if first_contact is None:
            # 没碰到球 (除非只剩黑8且已清台)
            foul_first_hit = True 
        elif first_contact not in player_targets:
            foul_first_hit = True

        # 3. 计分
        score = 0
        
        # 致命错误
        if cue_pocketed and eight_pocketed: return -500
        if cue_pocketed: return -100
        if eight_pocketed:
            # 如果是合法打黑8
            if len(player_targets) == 1 and player_targets[0] == "8":
                return 200 # 胜利
            else:
                return -500 # 输了

        # 犯规扣分
        if foul_first_hit: score -= 50
        
        # 进球得分
        score += len(own_pocketed) * 100
        score -= len(enemy_pocketed) * 30
        
        # 合法击打但没进球
        if score == 0 and not foul_first_hit:
            score = 10 # 鼓励合法触球

        return score

    # ================= 安全球与兜底 =================
    
    def _find_safety_shot(self, balls, targets, table):
        """寻找安全球 (轻碰己方球)"""
        # 复用 NewAgent 的逻辑，稍微优化参数
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        
        best_safety = None
        best_score = -float('inf')
        
        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            aim_vec = target_pos - cue_pos
            aim_vec[2] = 0
            dist = np.linalg.norm(aim_vec)
            if dist < 1e-6: continue
            
            # 检查路径
            if self._is_path_blocked(cue_pos, target_pos, balls, exclude_ids=['cue', target_id]):
                continue
                
            phi = np.degrees(np.arctan2(aim_vec[1], aim_vec[0])) % 360
            
            # 尝试极轻力度
            for V0 in [0.3, 0.6, 1.0]:
                action = {'V0': V0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
                # 简单模拟评估 (不写太复杂了，只要不犯规就行)
                # 这里假设只要能碰到且力度小就是好球
                score = 0
                if self._is_safe_sim(action, balls, table, targets):
                    score = 50 - V0 * 10 # 力度越小越好
                    if score > best_score:
                        best_score = score
                        best_safety = action
        
        return best_safety

    def _is_safe_sim(self, action, balls, table, targets):
        """快速验证安全球是否犯规"""
        curr_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        cue = pt.Cue(cue_ball_id="cue")
        cue.set_state(**action)
        shot = pt.System(table=table, balls=curr_balls, cue=cue)
        try:
            pt.simulate(shot, inplace=True)
            # 检查是否犯规
            # 简略：只要 cue 没进袋，且首撞是 target
            for bid, b in shot.balls.items():
                if b.state.s == 4: return False # 有球进袋就不算纯防守(或者算运气好)，这里保守点
            
            # 检查首撞
            for e in shot.events:
                 ids = list(e.ids) if hasattr(e, 'ids') else []
                 if 'cue' in ids and 'cushion' not in str(e.event_type).lower():
                     other = [x for x in ids if x != 'cue']
                     if other and other[0] in targets:
                         return True # 合法碰撞
                     elif other:
                         return False # 撞错球
            return False # 没撞到球
        except:
            return False

    def _random_safe_action(self, balls, table):
        """完全随机时的兜底"""
        # 复用 NewAgent 的逻辑
        for _ in range(10):
            action = self._random_action()
            # 简单检查黑8是否进袋
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            cue.set_state(**action)
            shot = pt.System(table=table, balls=sim_balls, cue=cue)
            try:
                pt.simulate(shot, inplace=True)
                eight_pocketed = False
                cue_pocketed = False
                for b in shot.balls.values():
                    if b.id == "8" and b.state.s == 4: eight_pocketed = True
                    if b.id == "cue" and b.state.s == 4: cue_pocketed = True
                if eight_pocketed or cue_pocketed: continue
                return action
            except:
                continue
        return self._random_action()
