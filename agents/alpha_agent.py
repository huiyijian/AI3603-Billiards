import os
import sys
import numpy as np
import torch
import copy
from .smart_agent import SmartAgent

# Add project root to path to import train.model
# File is in AI3603-Billiards/agents/alpha_agent.py
# Root is ../../..
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from train.model import ValueNetwork, extract_features, get_input_dim

class AlphaAgent(SmartAgent):
    """
    AlphaAgent: 基于 SmartAgent，但在 MCTS 评估阶段引入价值网络 (Value Network)
    来预估局面胜率，而非仅依赖单步回报。
    """
    def __init__(self, model_path=None):
        super().__init__()
        self.model = None
        self.device = torch.device("cpu") # Use CPU for inference to avoid overhead/compatibility issues
        
        # Load Model
        if model_path is None:
            # Default path
            model_path = os.path.join(project_root, 'train', 'value_net.pth')
            
        if os.path.exists(model_path):
            try:
                input_dim = get_input_dim()
                self.model = ValueNetwork(input_dim=input_dim)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"[AlphaAgent] Successfully loaded Value Network from {model_path}")
            except Exception as e:
                print(f"[AlphaAgent] Failed to load model: {e}")
                self.model = None
        else:
            print(f"[AlphaAgent] Model not found at {model_path}, running in pure SmartAgent mode.")

    def _run_mcts(self, candidates, balls, table, targets):
        """
        Override _run_mcts to include Value Network evaluation
        """
        # If no model, fallback to original MCTS
        if self.model is None:
            return super()._run_mcts(candidates, balls, table, targets)
            
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
                # 1. Immediate Reward (Rule-based)
                immediate_score = self._analyze_shot_reward(shot, last_state_snapshot, targets)
                
                # 2. Future Value (Network-based)
                # Check if game over condition in immediate result
                if immediate_score <= -400 or immediate_score >= 200:
                    # Terminal state (Win or Lose)
                    # analyze_shot_reward returns -500 for lose, 200 for win (roughly)
                    # We map this to full value
                    combined_score = immediate_score
                else:
                    # Non-terminal state, use Value Net
                    # Extract features from the resulting state (shot.balls)
                    # shot.balls is a dict of Ball objects with updated state
                    # We need to pass the correct 'my_targets' for the NEXT state?
                    # Actually, for the current player, the value of the resulting state 
                    # is "How good is this state for ME?".
                    # So we use 'targets' (my targets).
                    
                    try:
                        feats = extract_features(shot.balls, targets, table)
                        feats_tensor = torch.FloatTensor(feats).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            win_prob = self.model(feats_tensor).item()
                            
                        # Combine: Immediate + Future
                        # Immediate score is usually [-50, 100].
                        # Win Prob is [0, 1].
                        # We want to encourage high win prob.
                        # Let's say Win Prob 1.0 ~= +200 points.
                        
                        future_score = win_prob * 300.0 # Weighting factor
                        combined_score = immediate_score + future_score
                        
                    except Exception as e:
                        # Fallback if feature extraction fails
                        combined_score = immediate_score

                score = combined_score
            
            # Normalization (Updated range)
            # Min: -500, Max: ~400 (100 immediate + 300 future)
            norm_score = np.clip((score - (-500)) / 900.0, 0.0, 1.0)
            
            # Backprop
            N[idx] += 1
            Q[idx] += norm_score
            
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        
        # Record score
        for i in range(n_candidates):
            candidates[i]['score'] = avg_rewards[i] * 900 - 500
        
        print(f"[AlphaAgent] 最佳方案: {candidates[best_idx].get('shot_type', 'UNKNOWN')} "
              f"目标:{candidates[best_idx].get('target_ball')} "
              f"得分:{avg_rewards[best_idx]:.3f}")
