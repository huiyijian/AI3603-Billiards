import math
import numpy as np
import pooltool as pt
import copy
import random
from .agent import Agent

class NewAgent(Agent):
    """
    基于几何计算和物理模拟的增强型台球智能体。
    策略：
    1. 候选生成：遍历所有合法目标球和所有袋口，计算理想击球参数（Ghost Ball 算法）。
    2. 路径筛选：剔除切角过大或路径被阻挡的击球方案。
    3. 模拟优选：对筛选出的候选方案进行物理模拟（包含噪声扰动），选择期望回报最高的方案。
    4. 兜底策略：如果没有可行进攻方案，采用安全击球或随机击球。
    """
    
    def __init__(self):
        super().__init__()
        self.num_simulation_samples = 10  # 增加采样数
        self.max_candidates = 8          # 略微减少候选数以平衡时间
        self.pocket_ids = None

    def decision(self, balls=None, my_targets=None, table=None):
        """
        核心决策函数
        """
        if balls is None or table is None:
            return self._random_action()
            
        # 1. 确定目标球列表
        active_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not active_targets:
            if "8" in balls and balls["8"].state.s != 4:
                active_targets = ["8"]
            else:
                return self._random_action()

        # 2. 生成候选击球方案
        candidates = self._generate_candidates(balls, active_targets, table)
        
        if not candidates:
            print("[NewAgent] 无几何可行方案，尝试随机击球")
            return self._random_action()

        # 3. 模拟评估
        best_action = None
        best_score = -float('inf')
        
        # 对候选方案按几何容易度排序
        candidates.sort(key=lambda x: x['difficulty'])
        top_candidates = candidates[:self.max_candidates]

        print(f"[NewAgent] 评估 {len(top_candidates)} 个候选方案 (target={active_targets})")

        # 检查是否只剩黑8 (用于评分函数)
        # 注意：这里我们使用一个辅助标志，因为 my_targets 可能未更新
        is_clearing_black_eight = (len(active_targets) == 1 and active_targets[0] == "8")

        for cand in top_candidates:
            # 尝试优化力度 V0
            # 在基础 V0 附近搜索: [0.9*V0, 1.0*V0, 1.1*V0]
            base_V0 = cand['action']['V0']
            best_v_score = -float('inf')
            best_v = base_V0
            
            # 力度微调搜索
            for v_factor in [0.9, 1.0, 1.1]:
                test_cand = copy.deepcopy(cand)
                test_cand['action']['V0'] = base_V0 * v_factor
                test_cand['action']['V0'] = np.clip(test_cand['action']['V0'], 0.5, 7.0)
                
                score = self._evaluate_candidate(test_cand, balls, table, my_targets, is_clearing_black_eight)
                if score > best_v_score:
                    best_v_score = score
                    best_v = test_cand['action']['V0']
            
            # 使用最佳力度更新候选
            cand['action']['V0'] = best_v
            final_score = best_v_score

            if final_score > best_score:
                best_score = final_score
                best_action = cand['action']
                
            print(f"  - 目标:{cand['target_ball']} 袋口:{cand['pocket_id']} "
                  f"V0:{best_v:.1f} 切角:{cand['cut_angle']:.1f}° 分数:{final_score:.1f}")

        if best_action and best_score > -50: # 只有当分数不太差时才采用
            return best_action
        
        # 如果所有方案都很差（比如都会母球进袋，或者都犯规），尝试打一个安全球
        if best_action:
             print("[NewAgent] 所有方案分数较低，尝试寻找安全球")
             safety_action = self._find_safety_shot(balls, active_targets, table)
             if safety_action:
                 print("[NewAgent] 采用安全球策略")
                 return safety_action
             else:
                 print("[NewAgent] 安全球策略失败，仍采用最佳进攻方案")
                 return best_action

        # 如果连安全球也找不到（比如被完全锁死），尝试最后的随机兜底，但要避开黑8
        print("[NewAgent] 尝试随机兜底（避开黑8）")
        return self._random_safe_action(balls, table)

    def _random_safe_action(self, balls, table):
        """生成随机动作，但进行简单检查以避免直接输掉比赛（如打进黑8）"""
        for _ in range(10): # 尝试10次
            action = self._random_action()
            
            # 简单模拟一下
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            cue.set_state(**action)
            shot = pt.System(table=table, balls=sim_balls, cue=cue)
            
            try:
                pt.simulate(shot, inplace=True)
            except:
                continue
            
            # 检查是否犯规导致输掉比赛
            eight_pocketed = False
            cue_pocketed = False
            for ball in shot.balls.values():
                if ball.id == "8" and ball.state.s == 4:
                    eight_pocketed = True
                if ball.id == "cue" and ball.state.s == 4:
                    cue_pocketed = True
            
            # 如果黑8进袋（且没清台），或者白球+黑8进袋，这动作不能要
            # 这里简单点，只要黑8进袋就不要（除非我们不知道是否清台，但这里是兜底，保守点）
            if eight_pocketed:
                continue
                
            return action
            
        return self._random_action() # 实在不行就听天由命


    def _find_safety_shot(self, balls, targets, table):
        """寻找安全球：轻击，确保碰到己方球，且不犯规"""
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R

        best_safety = None
        best_safety_score = -float('inf')

        # 遍历目标球
        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 直接瞄准目标球中心
            aim_vector = target_pos - cue_pos
            aim_vector[2] = 0
            dist = np.linalg.norm(aim_vector)
            
            if dist < 1e-6: continue

            # 检查路径
            if self._is_path_blocked(cue_pos, target_pos, balls, exclude_ids=['cue', target_id]):
                continue

            phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0]))
            if phi < 0: phi += 360

            # 尝试不同的轻力度
            for V0 in [0.5, 1.0, 1.5]:
                action = {
                    'V0': V0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
                
                # 模拟评估
                score = self._evaluate_safety(action, balls, table, targets)
                if score > best_safety_score:
                    best_safety_score = score
                    best_safety = action
        
        if best_safety_score > -50: # 如果找到一个不太差的安全球
            return best_safety
        return None

    def _evaluate_safety(self, action, balls, table, my_targets):
        """评估安全球：目标是不犯规且不给对方机会"""
        total_score = 0
        num_samples = 5
        
        for _ in range(num_samples):
            noisy_action = copy.deepcopy(action)
            noisy_action['V0'] += np.random.normal(0, 0.1)
            noisy_action['phi'] += np.random.normal(0, 0.1)
            noisy_action['phi'] = noisy_action['phi'] % 360
            
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
            # 1. 必须碰到自己的球 (由 poolenv 规则，如果没碰到或先碰对方，会扣分)
            # 我们这里简单检查：shot.events 里是否有 cue 撞击 my_targets
            hit_own = False
            hit_wrong = False
            cue_pocketed = False
            
            for e in shot.events:
                ids = list(e.ids)
                if 'cue' in ids:
                    if 'cushion' in str(e.event_type).lower():
                        continue
                    if 'pocket' in str(e.event_type).lower():
                        cue_pocketed = True
                        break
                    # 检查撞到了谁
                    other = [i for i in ids if i != 'cue']
                    if other:
                        first_hit = other[0]
                        if first_hit in my_targets:
                            hit_own = True
                        else:
                            hit_wrong = True
                        break # 只看首撞
            
            if cue_pocketed:
                total_score -= 100
            elif hit_wrong:
                total_score -= 50
            elif not hit_own: # 没撞到任何球
                total_score -= 30
            else:
                # 合法撞击
                total_score += 10
                # 鼓励白球停得远？或者停在库边？暂时不加复杂策略
        
        return total_score / num_samples


    def _generate_candidates(self, balls, targets, table):
        """生成几何上可行的击球参数"""
        candidates = []
        cue_ball = balls['cue']
        cue_pos = cue_ball.state.rvw[0]
        R = cue_ball.params.R # 球半径

        for target_id in targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                # pocket.center 是袋口中心坐标 [x, y, z]
                pocket_pos = pocket.center
                
                # 1. 计算幽灵球位置 (Ghost Ball Position)
                # 目标球到袋口的向量
                to_pocket = pocket_pos - target_pos
                to_pocket[2] = 0 # 忽略高度差
                dist_to_pocket = np.linalg.norm(to_pocket)
                
                if dist_to_pocket < 1e-6:
                    continue
                    
                unit_to_pocket = to_pocket / dist_to_pocket
                
                # 幽灵球位置：目标球沿反方向退后 2R 距离
                ghost_pos = target_pos - unit_to_pocket * (2 * R)
                
                # 2. 计算击球向量（母球 -> 幽灵球）
                aim_vector = ghost_pos - cue_pos
                aim_vector[2] = 0
                dist_cue_to_ghost = np.linalg.norm(aim_vector)
                
                if dist_cue_to_ghost < 1e-6:
                    continue # 母球和幽灵球重合（实际上不可能，除非犯规）

                # 3. 计算切角 (Cut Angle)
                # 击球方向 与 目标球进袋方向 的夹角
                # cos(theta) = dot(v1, v2)
                dot_prod = np.dot(aim_vector / dist_cue_to_ghost, unit_to_pocket)
                # 限制在 [-1, 1] 避免数值误差
                dot_prod = np.clip(dot_prod, -1.0, 1.0)
                cut_angle_rad = np.arccos(dot_prod)
                cut_angle_deg = np.degrees(cut_angle_rad)
                
                # 过滤：切角大于 80 度极难打进
                if cut_angle_deg > 80:
                    continue

                # 4. 路径障碍检测 (简单版)
                # 检查 母球->幽灵球 路径上是否有其他球
                if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude_ids=['cue', target_id]):
                    continue
                
                # 检查 目标球->袋口 路径上是否有其他球
                if self._is_path_blocked(target_pos, pocket_pos, balls, exclude_ids=['cue', target_id]):
                    continue

                # 5. 计算击球参数 phi
                # pooltool 中 phi 是与 x 轴的夹角（弧度或度？API通常用度，但在 math 里用弧度）
                # 注意：pooltool 的 coordinate system. 
                # 通常：x, y. phi = arctan2(y, x)
                phi = np.degrees(np.arctan2(aim_vector[1], aim_vector[0]))
                if phi < 0:
                    phi += 360
                
                # 6. 估算力度 V0
                # 简单启发式：距离越远力度越大，切角越大力度稍大
                # 基础力度 1.5 m/s，每米增加 1.0 m/s
                total_dist = dist_cue_to_ghost + dist_to_pocket
                V0 = 1.0 + total_dist * 1.5
                # 稍微增加力度以确保进袋
                V0 *= 1.1 
                V0 = np.clip(V0, 0.5, 4.0) # 限制力度范围

                candidates.append({
                    'target_ball': target_id,
                    'pocket_id': pocket_id,
                    'action': {
                        'V0': V0,
                        'phi': phi,
                        'theta': 0.0, # 平击
                        'a': 0.0,
                        'b': 0.0
                    },
                    'cut_angle': cut_angle_deg,
                    'difficulty': cut_angle_deg + dist_cue_to_ghost * 10 # 距离也增加难度
                })
        
        return candidates

    def _is_path_blocked(self, start_pos, end_pos, balls, exclude_ids):
        """检测两点之间是否有障碍球"""
        # 简化模型：线段到点的距离
        path_vec = end_pos - start_pos
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6:
            return False
            
        path_dir = path_vec / path_len
        
        # 检查每个球
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4: # 已进袋
                continue
                
            ball_pos = ball.state.rvw[0]
            
            # 投影长度
            to_ball = ball_pos - start_pos
            proj_len = np.dot(to_ball, path_dir)
            
            # 球在这一段路径范围内 (前后预留一点余量)
            if 0 < proj_len < path_len:
                # 垂直距离
                # dist = ||vec - proj * dir||
                perp_vec = to_ball - proj_len * path_dir
                perp_dist = np.linalg.norm(perp_vec)
                
                # 如果垂直距离小于 2*R (两球直径)，则认为会碰撞
                # 这里稍微放宽一点判定，比如 1.9R，避免太保守
                if perp_dist < 2 * ball.params.R * 0.95:
                    return True
                    
        return False

    def _evaluate_candidate(self, candidate, balls, table, my_targets, is_clearing_black_eight=False):
        """对候选动作进行多次模拟评分"""
        base_action = candidate['action']
        total_score = 0
        
        success_count = 0
        
        for _ in range(self.num_simulation_samples):
            # 添加噪声 (模拟执行误差)
            # 这里的噪声标准差应该与环境一致或略大，以评估鲁棒性
            # 环境默认: V0=0.1, phi=0.1, theta=0.1, a=0.003, b=0.003
            noisy_action = copy.deepcopy(base_action)
            noisy_action['V0'] += np.random.normal(0, 0.1)
            noisy_action['phi'] += np.random.normal(0, 0.1)
            noisy_action['phi'] = noisy_action['phi'] % 360
            
            # 构建模拟系统
            curr_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            cue.set_state(**noisy_action)
            
            shot = pt.System(table=table, balls=curr_balls, cue=cue)
            
            try:
                pt.simulate(shot, inplace=True)
            except Exception:
                total_score -= 100 # 模拟失败惩罚
                continue
            
            # 评分
            score = self._calculate_score(shot, balls, my_targets, candidate['target_ball'], is_clearing_black_eight)
            total_score += score
            
            if score > 50: # 意味着进球了
                success_count += 1
        
        avg_score = total_score / self.num_simulation_samples
        
        # 额外奖励：成功率高的方案
        success_rate = success_count / self.num_simulation_samples
        avg_score += success_rate * 50 
        
        return avg_score

    def _calculate_score(self, shot, initial_balls, my_targets, intended_target, is_clearing_black_eight):
        """
        简化的评分函数
        """
        score = 0
        
        # 1. 检查进球
        pocketed_ids = []
        for bid, ball in shot.balls.items():
            if ball.state.s == 4 and initial_balls[bid].state.s != 4:
                pocketed_ids.append(bid)
        
        # 2. 关键事件判定
        cue_pocketed = "cue" in pocketed_ids
        eight_pocketed = "8" in pocketed_ids
        intended_pocketed = intended_target in pocketed_ids
        
        # 3. 犯规惩罚
        if cue_pocketed:
            return -100
        
        if eight_pocketed:
            # 如果不是只剩黑8，打进黑8判负
            if is_clearing_black_eight:
                return 200
            else:
                return -200
        
        # 4. 进球奖励
        if intended_pocketed:
            score += 100
            
        # 误进其他己方球
        for pid in pocketed_ids:
            if pid in my_targets and pid != intended_target:
                score += 50
            elif pid not in my_targets and pid != "cue" and pid != "8":
                score -= 20 # 进对方球
        
        # 5. 合法性检查 (是否碰到自己的球)
        # 从 shot.events 中检查首个碰撞
        # 简化处理：如果没进目标球，分数为0或负
        if not intended_pocketed and not pocketed_ids:
             # 简单的防守奖励：母球离所有对方球远？或者离袋口远？
             # 暂时不加复杂防守逻辑，避免为了防守而放弃进攻机会
             pass

        return score
