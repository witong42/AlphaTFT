import numpy as np
from typing import Dict, List, Tuple, Optional

class TFTEnvironment:
    """TFT Environment that simulates the game state and valid actions."""
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> Dict:
        """Reset the environment to initial state."""
        self.gold = 2
        self.level = 1
        self.health = 100
        self.round = 1
        self.shop = []
        self.bench = []
        self.board = []
        self.items = []
        self.streak = 0
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute one time step within the environment.
        
        Args:
            action: Dict containing action type and parameters
                   e.g. {'type': 'buy', 'unit_index': 0}
                   or {'type': 'move', 'from_pos': (0,0), 'to_pos': (1,0)}
        
        Returns:
            observation: Dict of current game state
            reward: Float of reward for the action
            done: Boolean whether the episode is finished
            info: Dict of additional information
        """
        action_type = action['type']
        
        if action_type == 'buy':
            reward = self._execute_buy(action['unit_index'])
        elif action_type == 'sell':
            reward = self._execute_sell(action['unit_index'])
        elif action_type == 'move':
            reward = self._execute_move(action['from_pos'], action['to_pos'])
        elif action_type == 'reroll':
            reward = self._execute_reroll()
        elif action_type == 'level_up':
            reward = self._execute_level_up()
        else:
            raise ValueError(f"Unknown action type: {action_type}")
            
        # Update game state
        self._update_state()
        
        # Check if game is done
        done = self.health <= 0
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self) -> Dict:
        """Get current game state observation."""
        return {
            'gold': self.gold,
            'level': self.level,
            'health': self.health,
            'round': self.round,
            'shop': self.shop,
            'bench': self.bench,
            'board': self.board,
            'items': self.items,
            'streak': self.streak
        }
    
    def _execute_buy(self, unit_index: int) -> float:
        """Execute buy action."""
        if unit_index >= len(self.shop) or self.gold < self.shop[unit_index]['cost']:
            return -0.1  # Invalid action penalty
        
        unit = self.shop[unit_index]
        if len(self.bench) >= 9:  # Max bench size
            return -0.1
            
        self.gold -= unit['cost']
        self.bench.append(unit)
        self.shop.pop(unit_index)
        
        return 0.1  # Small positive reward for valid action
    
    def _execute_sell(self, unit_index: int) -> float:
        """Execute sell action."""
        if unit_index >= len(self.bench):
            return -0.1
            
        unit = self.bench[unit_index]
        self.gold += unit['cost']
        self.bench.pop(unit_index)
        
        return 0.05
    
    def _execute_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Execute move action between bench and board."""
        # Implementation depends on specific game rules
        return 0.0
    
    def _execute_reroll(self) -> float:
        """Execute shop reroll action."""
        if self.gold < 2:
            return -0.1
            
        self.gold -= 2
        self._refresh_shop()
        return 0.0
    
    def _execute_level_up(self) -> float:
        """Execute level up action."""
        cost = 4  # Base cost, should scale with level
        if self.gold < cost or self.level >= 9:
            return -0.1
            
        self.gold -= cost
        self.level += 1
        return 0.2
    
    def _update_state(self):
        """Update game state after each action."""
        # Update gold income, round, etc.
        pass
        
    def _refresh_shop(self):
        """Refresh shop with new units."""
        # Implementation should consider unit pool, probabilities, etc.
        pass
        
    def get_valid_actions(self) -> List[Dict]:
        """Get list of valid actions in current state."""
        valid_actions = []
        
        # Buy actions
        for i in range(len(self.shop)):
            if self.gold >= self.shop[i]['cost'] and len(self.bench) < 9:
                valid_actions.append({'type': 'buy', 'unit_index': i})
        
        # Sell actions
        for i in range(len(self.bench)):
            valid_actions.append({'type': 'sell', 'unit_index': i})
        
        # Reroll action
        if self.gold >= 2:
            valid_actions.append({'type': 'reroll'})
        
        # Level up action
        if self.gold >= 4 and self.level < 9:
            valid_actions.append({'type': 'level_up'})
        
        return valid_actions
