import numpy as np

def to_2d(p):
    """强制将坐标转换为2D numpy数组"""
    return np.array(p)[:2]

def get_distance(p1, p2):
    """计算两点欧几里得距离 (2D)"""
    return np.linalg.norm(to_2d(p1) - to_2d(p2))

def calculate_shot_phi(cue_pos, target_pos):
    """计算从 cue_pos 指向 target_pos 的角度 phi (弧度)"""
    p1 = to_2d(cue_pos)
    p2 = to_2d(target_pos)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def calculate_ghost_ball_pos(target_pos, pocket_pos, ball_radius):
    """
    计算幻影球位置 (2D计算，返回3D格式以便兼容)
    """
    t_vec = to_2d(target_pos)
    p_vec = to_2d(pocket_pos)
    
    # 从袋口指向目标球的向量
    vec = t_vec - p_vec
    
    # 单位化
    norm = np.linalg.norm(vec)
    if norm == 0: return target_pos # 避免除0
    unit_vec = vec / norm
    
    # 幻影球在目标球后方 2 * r 处
    ghost_2d = t_vec + unit_vec * (2 * ball_radius)
    
    # 补全Z轴 (假设球在台面上，Z通常很小，保持原target的Z即可)
    z = target_pos[2] if len(target_pos) > 2 else 0
    return np.array([ghost_2d[0], ghost_2d[1], z])

def is_path_blocked(start_pos, end_pos, obstacle_positions, ball_radius, clearance=0.0):
    """
    检测路径是否被阻挡 (强制2D投影检测)
    """
    p1 = to_2d(start_pos)
    p2 = to_2d(end_pos)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0: return False
    dir_vec = vec / length
    
    # 检测每个障碍球
    for obs_pos in obstacle_positions:
        op = to_2d(obs_pos)
        
        # 排除起点和终点附近的球（比如自己就是障碍物的情况）
        if np.linalg.norm(op - p1) < 0.01 or np.linalg.norm(op - p2) < 0.01:
            continue
            
        # 投影点计算
        t = np.dot(op - p1, dir_vec)
        
        # 障碍球必须在路径段之间
        if 0 < t < length:
            closest_point = p1 + t * dir_vec
            dist = np.linalg.norm(op - closest_point)
            
            # 判定阈值
            if dist < (2 * ball_radius + clearance):
                return True
    return False

def is_cut_angle_safe(cue_pos, target_pos, pocket_pos, limit=80):
    """
    判断切球角度是否过大 (2D)
    """
    # 向量1: 目标球 -> 袋口
    v_target_pocket = to_2d(pocket_pos) - to_2d(target_pos)
    # 向量2: 白球 -> 目标球
    v_cue_target = to_2d(target_pos) - to_2d(cue_pos)
    
    norm1 = np.linalg.norm(v_target_pocket)
    norm2 = np.linalg.norm(v_cue_target)
    
    if norm1 == 0 or norm2 == 0: return False
    
    cos_angle = np.dot(v_target_pocket, v_cue_target) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_deg = np.degrees(np.arccos(cos_angle))
    return abs(angle_deg) < limit