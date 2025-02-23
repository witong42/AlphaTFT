import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from ..models.tft_model import TFTAgent
from ..environment.tft_env import TFTEnvironment
import ast

class TFTTrainer:
    """Trainer class that handles both supervised and reinforcement learning."""
    
    def __init__(self, agent: TFTAgent, env: TFTEnvironment):
        self.agent = agent
        self.env = env
        
    def load_expert_data(self, data_path: str) -> Tuple[List[Dict], List[Dict], List[float]]:
        """Load and preprocess expert game data.
        
        Args:
            data_path: Path to the processed matches CSV file
            
        Returns:
            states: List of game states
            actions: List of actions taken
            values: List of state values (estimated from game outcome)
        """
        # Load data
        df = pd.read_csv(data_path)
        
        states = []
        actions = []
        values = []
        
        # Process each match
        for _, row in df.iterrows():
            # Convert row data to state dict
            state = {
                'gold': 50,  # Default starting gold
                'level': row['level'],
                'health': 100,  # Default starting health
                'round': 1,  # Start from round 1
                'streak': 0,  # Start with no streak
                'shop': [],  # Empty shop at start
                'bench': [],  # Start with empty bench
                'board': self._parse_units(row['units']),  # Parse final board state
                'items': []  # Start with no items
            }
            
            # For now, we'll use a simple "buy all units" sequence as our action set
            # In the future, we can extract actual actions from game data
            board_units = self._parse_units(row['units'])
            for i, unit in enumerate(board_units):
                actions.append({
                    'type': 'buy',
                    'unit_index': i % 5  # Simulate buying from shop slots 0-4
                })
            
            # Calculate value based on placement (8th = 0.0, 1st = 1.0)
            value = (8 - row['placement']) / 7.0
            
            states.append(state)
            values.append(value)
        
        return states, actions, values
    
    def train_supervised(self, expert_data_path: str, num_epochs: int = 10,
                        batch_size: int = 32) -> List[float]:
        """Train agent using supervised learning on expert data."""
        states, actions, values = self.load_expert_data(expert_data_path)
        
        # Convert to tensors
        states_tensor = torch.stack([
            self.agent._preprocess_state(state).squeeze()
            for state in states
        ])
        
        actions_tensor = torch.tensor([
            self.agent._action_to_index(action)
            for action in actions
        ], dtype=torch.long)
        
        values_tensor = torch.tensor(values, dtype=torch.float32)
        
        # Training loop
        losses = []
        num_batches = len(states) // batch_size
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(states))
            states_tensor = states_tensor[indices]
            actions_tensor = actions_tensor[indices]
            values_tensor = values_tensor[indices]
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                batch_states = states_tensor[start_idx:end_idx]
                batch_actions = actions_tensor[start_idx:end_idx]
                batch_values = values_tensor[start_idx:end_idx]
                
                # Forward pass
                policy_logits, value_preds = self.agent.network(batch_states)
                
                # Policy loss (cross entropy)
                policy_loss = torch.nn.functional.cross_entropy(
                    policy_logits, batch_actions
                )
                
                # Value loss (MSE)
                value_loss = torch.nn.functional.mse_loss(
                    value_preds.squeeze(), batch_values
                )
                
                # Combined loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward pass
                self.agent.optimizer.zero_grad()
                total_loss.backward()
                self.agent.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_epoch_loss = epoch_loss / num_batches
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return losses
    
    def _parse_units(self, units_str: str) -> List[Dict]:
        """Parse units string into list of unit dicts."""
        try:
            units = ast.literal_eval(units_str)
            return [
                {
                    'character_id': unit['character_id'],
                    'position': (i // 7, i % 7)  # Assign positions in a grid
                }
                for i, unit in enumerate(units)
            ]
        except:
            return []

if __name__ == "__main__":
    # Example usage
    STATE_DIM = 256  # Depends on your state representation
    ACTION_DIM = 32  # Depends on your action space
    
    trainer = TFTTrainer(TFTAgent(STATE_DIM, ACTION_DIM), TFTEnvironment())
    
    # Load expert data
    trainer.load_expert_data("data/processed_matches_20250223.csv")
    
    # First train with supervised learning
    print("Starting supervised learning...")
    sl_losses = trainer.train_supervised("data/processed_matches_20250223.csv", num_epochs=1000)
    
    # Then fine-tune with reinforcement learning
    # rl_rewards = trainer.train_reinforcement(num_episodes=1000)
    
    # Save the trained agent
    # trainer.save_agent("models/tft_agent.pt")
