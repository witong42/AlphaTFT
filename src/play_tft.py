import sys
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import urllib3
import subprocess
import requests
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from src.models.tft_model import TFTAgent
from src.environment.tft_env import TFTEnvironment
from src.data.riot_api import RiotAPI
from src.ocr.screen_capture import TFTScreenCapture

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TFTGameInterface:
    """Interface for playing TFT using our trained model."""
    
    def __init__(self, model_path: str):
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with same dimensions as training
        STATE_DIM = 5 + (5 + 9 + 28) * 61 + 30
        ACTION_DIM = 5 + 9 + (37 * 37) + 2
        
        self.agent = TFTAgent(STATE_DIM, ACTION_DIM, device=str(self.device))
        self.agent.load(model_path)
        
        # Initialize environment
        self.env = TFTEnvironment()
        
        # Initialize Riot API client
        self.riot_api = RiotAPI()
        
        # Initialize screen capture
        self.screen_capture = TFTScreenCapture()
        
        # For debugging
        self.debug_mode = True
        
    def is_tft_running(self) -> bool:
        """Check if TFT is running using system commands."""
        try:
            if sys.platform == "darwin":  # macOS
                output = subprocess.check_output(["ps", "-A"]).decode()
                return "League of Legends" in output or "TFT" in output
            elif sys.platform == "win32":  # Windows
                output = subprocess.check_output(["tasklist"]).decode()
                return "League of Legends.exe" in output or "TFT.exe" in output
            return False
        except:
            return False
            
    def is_in_game(self) -> bool:
        """Check if we're in an active game."""
        try:
            response = requests.get(
                "https://127.0.0.1:2999/liveclientdata/gamestats",
                verify=False,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
            
    def get_current_game_state(self) -> Optional[Dict]:
        """Get the current game state from the TFT client."""
        if not self.is_tft_running():
            logging.info("TFT is not running. Please start the game first.")
            return None
            
        try:
            # Try different ports and endpoints that TFT might use
            ports = [2999, 2999, 2999]
            endpoints = [
                'allgamedata',
                'playerdata',
                'gamestats'
            ]
            
            game_data = None
            for port in ports:
                for endpoint in endpoints:
                    try:
                        url = f"https://127.0.0.1:{port}/liveclientdata/{endpoint}"
                        print(f"\nTrying endpoint: {url}")
                        response = requests.get(url, verify=False, timeout=2)
                        print(f"Status code: {response.status_code}")
                        
                        if response.status_code == 200:
                            data = response.json()
                            print(f"\nData from {endpoint}:")
                            print(json.dumps(data, indent=2))
                            game_data = data
                            break
                    except Exception as e:
                        print(f"Error with {endpoint} on port {port}: {str(e)}")
                        continue
                        
                if game_data:
                    break
            
            if not game_data:
                print("Could not get game data from any endpoint")
                return None
            
            # Try to extract state data based on the endpoint that worked
            state = {
                'gold': 0,
                'level': 1,
                'health': 100,
                'round': 1,
                'streak': 0,
                'shop': [],
                'bench': [],
                'board': [],
                'items': []
            }
            
            # Update state based on which data we got
            if 'activePlayer' in game_data:
                active_player = game_data['activePlayer']
                state.update({
                    'gold': active_player.get('gold', state['gold']),
                    'level': active_player.get('level', state['level']),
                    'health': active_player.get('health', state['health'])
                })
            elif 'scores' in game_data:
                scores = game_data['scores']
                state.update({
                    'gold': scores.get('gold', state['gold']),
                    'level': scores.get('level', state['level']),
                    'health': scores.get('health', state['health'])
                })
            
            if 'gameData' in game_data:
                game_stats = game_data['gameData']
                state['round'] = game_stats.get('gameTime', 0) // 30 + 1
            
            # Get OCR data if available
            try:
                screen_state = self.screen_capture.get_game_state()
                state.update({
                    'shop': screen_state['shop'],
                    'bench': screen_state['bench'],
                    'board': screen_state['board'],
                    'items': screen_state['items']
                })
            except Exception as e:
                logging.error(f"Error getting screen state: {e}")
            
            print("\nFinal processed state:")
            print(json.dumps(state, indent=2))
            
            return state
            
        except Exception as e:
            logging.error(f"Unexpected error in get_current_game_state: {e}")
            return None
            
    def suggest_action(self, state: Dict) -> Dict:
        """Get model's suggested action for the current state."""
        try:
            # Update environment state
            self.env.gold = state['gold']
            self.env.level = state['level']
            self.env.health = state['health']
            self.env.round = state['round']
            self.env.streak = state['streak']
            self.env.shop = state['shop']
            self.env.bench = state['bench']
            self.env.board = state['board']
            self.env.items = state['items']
            
            # Get valid actions from environment
            valid_actions = self.env.get_valid_actions()
            
            # Get model's action
            action = self.agent.get_action(state, valid_actions)
            
            return action
        except Exception as e:
            logging.error(f"Error in suggest_action: {e}")
            # Return a safe default action
            return {'type': 'reroll'}
        
    def play_game(self):
        """Main loop for playing a TFT game."""
        print("Waiting for TFT game to start...")
        print("Make sure you:")
        print("1. Have TFT running")
        print("2. Are in an active game")
        print("3. Have the game window focused")
        print("\nPress Ctrl+C to exit\n")
        
        consecutive_errors = 0
        while True:
            try:
                if not self.is_tft_running():
                    print("TFT is not running. Please start TFT first.")
                    time.sleep(5)
                    continue
                    
                if not self.is_in_game():
                    if consecutive_errors == 0:  # Only print once
                        print("Waiting for game to start...")
                    consecutive_errors += 1
                    time.sleep(2)
                    continue
                
                state = self.get_current_game_state()
                
                if state is None:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        time.sleep(5)  # Wait longer between retries
                    else:
                        time.sleep(2)
                    continue
                
                if self.debug_mode:
                    logging.debug(f"Current state: {state}")
                
                consecutive_errors = 0  # Reset error counter on success
                
                # Get model's suggested action
                action = self.suggest_action(state)
                
                # Display suggestion to user
                self._display_suggestion(state, action)
                
                # Wait before next suggestion
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error(f"Unexpected error in play_game: {e}")
                time.sleep(2)
            
    def _parse_round(self, game_time: float) -> int:
        """Convert game time to round number."""
        try:
            return max(1, int(float(game_time) / 30))
        except:
            return 1
        
    def _parse_shop(self, shop_data: List) -> List[Dict]:
        """Parse shop data from game client."""
        try:
            return [
                {'character_id': str(unit.get('characterId', 'unknown'))}
                for unit in shop_data
                if isinstance(unit, dict)
            ]
        except:
            return []
        
    def _parse_bench(self, bench_data: List) -> List[Dict]:
        """Parse bench data from game client."""
        try:
            return [
                {'character_id': str(unit.get('characterId', 'unknown'))}
                for unit in bench_data
                if isinstance(unit, dict)
            ]
        except:
            return []
        
    def _parse_board(self, board_data: List) -> List[Dict]:
        """Parse board data from game client."""
        try:
            return [
                {
                    'character_id': str(unit.get('characterId', 'unknown')),
                    'position': (
                        int(unit.get('position', {}).get('row', 0)),
                        int(unit.get('position', {}).get('col', 0))
                    )
                }
                for unit in board_data
                if isinstance(unit, dict)
            ]
        except:
            return []
        
    def _parse_items(self, items_data: List) -> List[Dict]:
        """Parse items data from game client."""
        try:
            return [
                {'id': str(item.get('id', 'unknown'))}
                for item in items_data
                if isinstance(item, dict)
            ]
        except:
            return []
        
    def _display_suggestion(self, state: Dict, action: Dict):
        """Display the model's suggested action to the user."""
        print("\n=== Current State ===")
        print(f"Gold: {state['gold']}")
        print(f"Level: {state['level']}")
        print(f"Health: {state['health']}")
        print(f"Round: {state['round']}")
        
        if self.debug_mode:
            print("\nBoard:")
            print(f"Units on board: {len(state['board'])}")
            for unit in state['board']:
                print(f"  {unit['character_id']} at position {unit['position']}")
            
            print("\nBench:")
            print(f"Units on bench: {len(state['bench'])}")
            for unit in state['bench']:
                print(f"  {unit['character_id']}")
            
            print("\nShop:")
            print(f"Units in shop: {len(state['shop'])}")
            for unit in state['shop']:
                print(f"  {unit['character_id']}")
        
        print("\n=== Suggested Action ===")
        if action['type'] == 'buy':
            print(f"Buy unit from shop slot {action['unit_index']}")
        elif action['type'] == 'sell':
            print(f"Sell unit from bench slot {action['unit_index']}")
        elif action['type'] == 'move':
            print(f"Move unit from {action['from_pos']} to {action['to_pos']}")
        elif action['type'] == 'reroll':
            print("Reroll shop")
        elif action['type'] == 'level_up':
            print("Level up")
        
        print("\n" + "="*30 + "\n")

def main():
    # Path to saved model
    model_path = "models/tft_agent.pt"
    
    # Create game interface
    game_interface = TFTGameInterface(model_path)
    
    # Start playing
    game_interface.play_game()

if __name__ == "__main__":
    main()
