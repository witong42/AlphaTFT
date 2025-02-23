import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Union

class TFTNetwork(nn.Module):
    """Neural network for TFT agent that outputs both policy and value."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(TFTNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy head (outputs action probabilities)
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value head (outputs state value estimation)
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = self.shared(state)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

class TFTAgent:
    """Agent that combines supervised learning and reinforcement learning."""
    
    # Action type mappings
    ACTION_TYPES = {
        'buy': 0,
        'sell': 1,
        'move': 2,
        'reroll': 3,
        'level_up': 4
    }
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.network = TFTNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
        # Define max values for normalization
        self.max_gold = 100
        self.max_level = 9
        self.max_health = 100
        self.max_round = 40
        self.max_streak = 10
        self.board_size = (4, 7)  # TFT board is 4x7
        self.bench_size = 9
        self.shop_size = 5
        
    def get_action(self, state: Dict, valid_actions: List[Dict]) -> Dict:
        """Get action from current policy with valid action masking."""
        state_tensor = self._preprocess_state(state)
        
        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor)
            
            # Mask invalid actions
            action_mask = torch.zeros_like(policy_logits)
            for action in valid_actions:
                action_idx = self._action_to_index(action)
                action_mask[action_idx] = 1
            
            masked_logits = policy_logits + (action_mask + 1e-10).log()
            action_probs = F.softmax(masked_logits, dim=-1)
            
            # Sample action from probability distribution
            action_index = torch.multinomial(action_probs, 1).item()
            
        return self._index_to_action(action_index)
    
    def _preprocess_state(self, state: Dict) -> torch.Tensor:
        """Convert state dict to tensor.
        
        State features:
        1. Scalar features (normalized):
           - Gold
           - Level
           - Health
           - Round
           - Streak
        2. Shop units (one-hot encoded)
        3. Bench units (one-hot encoded)
        4. Board units (one-hot encoded)
        5. Items (one-hot encoded)
        """
        # Normalize scalar features
        scalar_features = torch.tensor([
            state['gold'] / self.max_gold,
            state['level'] / self.max_level,
            state['health'] / self.max_health,
            state['round'] / self.max_round,
            state['streak'] / self.max_streak
        ], dtype=torch.float32)
        
        # Encode shop units
        shop_features = self._encode_units(state['shop'], self.shop_size)
        
        # Encode bench units
        bench_features = self._encode_units(state['bench'], self.bench_size)
        
        # Encode board units
        board_features = self._encode_board(state['board'])
        
        # Encode items
        item_features = self._encode_items(state['items'])
        
        # Concatenate all features
        state_tensor = torch.cat([
            scalar_features,
            shop_features.flatten(),
            bench_features.flatten(),
            board_features.flatten(),
            item_features
        ])
        
        return state_tensor.unsqueeze(0).to(self.device)
    
    def _encode_units(self, units: List[Dict], max_size: int) -> torch.Tensor:
        """Encode units into one-hot tensor."""
        # Assuming we have a fixed set of possible units
        num_unit_types = 60  # Total number of possible units
        encoding = torch.zeros((max_size, num_unit_types + 1))  # +1 for empty slot
        
        for i, unit in enumerate(units[:max_size]):
            if unit:
                unit_id = self._get_unit_id(unit['character_id'])
                encoding[i, unit_id] = 1
            else:
                encoding[i, -1] = 1  # Mark as empty slot
                
        return encoding
    
    def _encode_board(self, board: List[Dict]) -> torch.Tensor:
        """Encode board state into tensor."""
        num_unit_types = 60
        encoding = torch.zeros((self.board_size[0], self.board_size[1], num_unit_types + 1))
        
        for unit in board:
            pos = unit['position']
            unit_id = self._get_unit_id(unit['character_id'])
            encoding[pos[0], pos[1], unit_id] = 1
            
        return encoding
    
    def _encode_items(self, items: List[Dict]) -> torch.Tensor:
        """Encode items into one-hot tensor."""
        num_item_types = 30  # Total number of possible items
        encoding = torch.zeros(num_item_types)
        
        for item in items:
            item_id = self._get_item_id(item['id'])
            encoding[item_id] = 1
            
        return encoding
    
    def _get_unit_id(self, character_id: str) -> int:
        """Convert character ID to numeric ID."""
        # Implementation depends on your unit ID system
        # For now, just hash the string to an integer in range
        return hash(character_id) % 60
    
    def _get_item_id(self, item_id: str) -> int:
        """Convert item ID to numeric ID."""
        # Implementation depends on your item ID system
        return hash(item_id) % 30
    
    def _action_to_index(self, action: Dict) -> int:
        """Convert action dict to index.
        
        Action space:
        - Buy: shop_size actions
        - Sell: bench_size actions
        - Move: (bench_size + board_size) * (bench_size + board_size) actions
        - Reroll: 1 action
        - Level up: 1 action
        """
        base_index = 0
        action_type = action['type']
        
        if action_type == 'buy':
            return base_index + action['unit_index']
        
        base_index += self.shop_size  # After buy actions
        
        if action_type == 'sell':
            return base_index + action['unit_index']
            
        base_index += self.bench_size  # After sell actions
        
        if action_type == 'move':
            total_positions = self.bench_size + (self.board_size[0] * self.board_size[1])
            from_idx = self._position_to_index(action['from_pos'])
            to_idx = self._position_to_index(action['to_pos'])
            return base_index + (from_idx * total_positions + to_idx)
            
        base_index += (self.bench_size + (self.board_size[0] * self.board_size[1])) ** 2  # After move actions
        
        if action_type == 'reroll':
            return base_index
            
        base_index += 1  # After reroll action
        
        if action_type == 'level_up':
            return base_index
            
        raise ValueError(f"Unknown action type: {action_type}")
    
    def _index_to_action(self, index: int) -> Dict:
        """Convert index to action dict."""
        # Buy actions
        if index < self.shop_size:
            return {'type': 'buy', 'unit_index': index}
        
        index -= self.shop_size
        
        # Sell actions
        if index < self.bench_size:
            return {'type': 'sell', 'unit_index': index}
            
        index -= self.bench_size
        
        # Move actions
        total_positions = self.bench_size + (self.board_size[0] * self.board_size[1])
        if index < total_positions * total_positions:
            from_idx = index // total_positions
            to_idx = index % total_positions
            return {
                'type': 'move',
                'from_pos': self._index_to_position(from_idx),
                'to_pos': self._index_to_position(to_idx)
            }
            
        index -= total_positions * total_positions
        
        # Reroll action
        if index == 0:
            return {'type': 'reroll'}
            
        # Level up action
        return {'type': 'level_up'}
    
    def _position_to_index(self, pos: Tuple[int, int]) -> int:
        """Convert board position to index."""
        if isinstance(pos, int):  # Bench position
            return pos
        # Board position
        return self.bench_size + (pos[0] * self.board_size[1] + pos[1])
    
    def _index_to_position(self, index: int) -> Union[int, Tuple[int, int]]:
        """Convert index to position."""
        if index < self.bench_size:  # Bench position
            return index
        # Board position
        index -= self.bench_size
        return (index // self.board_size[1], index % self.board_size[1])
    
    def save(self, path: str):
        """Save model to path."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model from path."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
