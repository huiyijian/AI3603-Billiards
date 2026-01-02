"""
evaluate_parallel.py - 并行化 Agent 评估脚本

功能：
- 使用多进程并行运行游戏，加速评估
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 修改 main() 函数中的 agent_a_name 和 agent_b_name
2. 调整 n_games
3. 运行脚本
"""

import sys
import os
import multiprocessing
import collections
import time

# 修复 pooltool 导入路径
# 假设脚本位于 AI3603-Billiards 目录下，而 pooltool 包在 ../pooltool/pooltool
# 所以我们需要添加 ../pooltool 到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pooltool')))

# 也可以添加当前目录到 path，防止直接运行时找不到模块
sys.path.append(os.path.dirname(__file__))

# 延迟导入，防止在主进程导入时出错，但在 Windows 上这会在每个子进程中执行
try:
    from utils import set_random_seed
    from poolenv import PoolEnv
    from agents import BasicAgent, BasicAgentPro, NewAgent, AdvancedAgent, SmartAgent, AlphaAgent
except ImportError as e:
    # 第一次运行时可能会因为 path 问题报错，但在子进程中 path 已经设置好了
    pass

def play_single_game(game_id, agent_a_cls_name, agent_b_cls_name, seed):
    """
    运行单场比赛
    """
    # 重新导入以确保在子进程中可用
    from utils import set_random_seed
    from poolenv import PoolEnv
    from agents import BasicAgent, BasicAgentPro, NewAgent, AdvancedAgent, SmartAgent, AlphaAgent
    
    # 设置随机种子
    set_random_seed(enable=False, seed=seed)
    
    env = PoolEnv()
    
    # 映射类名到类
    agent_map = {
        'BasicAgent': BasicAgent,
        'BasicAgentPro': BasicAgentPro,
        'NewAgent': NewAgent,
        'AdvancedAgent': AdvancedAgent,
        'SmartAgent': SmartAgent,
        'AlphaAgent': AlphaAgent
    }
    
    try:
        agent_a = agent_map[agent_a_cls_name]()
        agent_b = agent_map[agent_b_cls_name]()
    except KeyError:
        return {
            'game_id': game_id,
            'error': f"Unknown agent class: {agent_a_cls_name} or {agent_b_cls_name}"
        }
    
    players = [agent_a, agent_b]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    # 比赛初始化
    env.reset(target_ball=target_ball_choice[game_id % 4])
    
    # 确保交替先手
    # game_id % 2 == 0: players[0] (agent_a) 先手
    # game_id % 2 == 1: players[1] (agent_b) 先手
    
    # 游戏循环
    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        
        if player == 'A':
            action = players[game_id % 2].decision(*obs)
        else:
            action = players[(game_id + 1) % 2].decision(*obs)
            
        env.take_shot(action)
        done, info = env.get_done()
        
        if done:
            return {
                'game_id': game_id,
                'winner': info['winner'],
                'reason': info.get('reason', 'UNKNOWN'),
                'agent_a_name': agent_a_cls_name,
                'agent_b_name': agent_b_cls_name
            }

def main():
    # 参数设置
    n_games = 120
    # 使用所有可用核心以获得最快速度
    n_processes = multiprocessing.cpu_count() 
    
    agent_a_name = 'AlphaAgent'
    agent_b_name = 'BasicAgent'
    
    print(f"开始并行评估: {n_games} 局, 使用 {n_processes} 个进程")
    print(f"Agent A: {agent_a_name}, Agent B: {agent_b_name}")
    
    start_time = time.time()
    
    # 准备任务
    tasks = [(i, agent_a_name, agent_b_name, 42 + i) for i in range(n_games)]
    
    results_list = []
    
    # 使用进程池
    # 注意：Windows下必须在 if __name__ == "__main__": 块中运行
    with multiprocessing.Pool(processes=n_processes) as pool:
        # 使用 starmap_async 以便我们可以跟踪进度 (这里简化为 starmap)
        # 为了进度条，我们可以分批或者使用 imap
        for i, res in enumerate(pool.starmap(play_single_game, tasks)):
            if 'error' in res:
                print(f"Game {res['game_id']} failed: {res['error']}")
                continue
                
            results_list.append(res)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(f"已完成 {i + 1}/{n_games} 局 (速度: {speed:.2f} 局/秒)")

    end_time = time.time()
    print(f"\n评估完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 统计结果
    final_results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    loss_reasons = collections.defaultdict(lambda: collections.defaultdict(int))
    
    for res in results_list:
        if 'error' in res: continue
        
        game_id = res['game_id']
        winner = res['winner']
        reason = res['reason']
        
        # 统计胜负
        if winner == 'SAME':
            final_results['SAME'] += 1
        elif winner == 'A':
            if game_id % 2 == 0: # A 是 agent_a
                final_results['AGENT_A_WIN'] += 1
            else: # A 是 agent_b
                final_results['AGENT_B_WIN'] += 1
        else: # winner == 'B'
            if game_id % 2 == 0: # B 是 agent_b
                final_results['AGENT_B_WIN'] += 1
            else: # B 是 agent_a
                final_results['AGENT_A_WIN'] += 1
                
        # 统计原因
        # 确定谁赢了 (Agent Name)
        winner_agent_name = "DRAW"
        current_player_a_is_agent_a = (game_id % 2 == 0)
        
        if winner == 'A':
            winner_agent_name = agent_a_name if current_player_a_is_agent_a else agent_b_name
        elif winner == 'B':
            winner_agent_name = agent_b_name if current_player_a_is_agent_a else agent_a_name
            
        if winner != 'SAME':
            loss_reasons[winner_agent_name][reason] += 1
            
            # 记录输家原因
            loser_agent_name = agent_b_name if winner_agent_name == agent_a_name else agent_a_name
            loss_reasons[loser_agent_name][f"LOST_BY_{reason}"] += 1

    # 计算总分
    final_results['AGENT_A_SCORE'] = final_results['AGENT_A_WIN'] * 1 + final_results['SAME'] * 0.5
    final_results['AGENT_B_SCORE'] = final_results['AGENT_B_WIN'] * 1 + final_results['SAME'] * 0.5

    print("\n最终结果：", final_results)
    print("\n详细结束原因统计:")
    for agent_name, reasons in loss_reasons.items():
        print(f"\n{agent_name}:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    # Windows 下 multiprocessing 必须使用 freeze_support (如果打包的话) 
    # 或者直接运行
    multiprocessing.freeze_support()
    main()
