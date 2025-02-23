import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from riot_api import RiotAPI
import traceback

class DataCollector:
    def __init__(self, data_dir: str = None):
        self.api = RiotAPI()
        if data_dir is None:
            # Get the absolute path to the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.raw_dir, exist_ok=True)

    def _save_match(self, match_data: Dict[str, Any], match_id: str) -> None:
        """Save match data to a JSON file."""
        filename = os.path.join(self.raw_dir, f"{match_id}.json")
        with open(filename, 'w') as f:
            json.dump(match_data, f)

    def collect_challenger_matches(self, matches_per_player: int = 20, max_players: int = 30) -> None:
        """Collect matches from challenger players.
        
        Args:
            matches_per_player: Number of matches to collect per player
            max_players: Maximum number of players to process
        """
        try:
            # Get challenger players
            challenger_data = self.api.get_challenger_players()
            players = challenger_data.get('entries', [])
            total_players = len(players)
            players = players[:max_players]  # Limit to max_players
            print(f"Found {total_players} challenger players, processing first {len(players)}")
            
            if not players:
                print("No players found in challenger data:")
                print(json.dumps(challenger_data, indent=2))
                return

            matches_collected = 0
            for idx, player in enumerate(players):
                try:
                    # Get summoner info using summoner ID
                    summoner_id = player.get('summonerId')
                    if not summoner_id:
                        print(f"No summoner ID found for player {idx}:")
                        print(json.dumps(player, indent=2))
                        continue
                        
                    print(f"\nProcessing player {idx + 1}/{len(players)} (ID: {summoner_id})")
                    summoner_info = self.api.get_summoner_by_id(summoner_id)
                    
                    if not summoner_info:
                        print(f"Could not get summoner info for ID {summoner_id}")
                        continue
                        
                    puuid = summoner_info.get('puuid')
                    if not puuid:
                        print(f"No PUUID found in summoner info for ID {summoner_id}:")
                        print(json.dumps(summoner_info, indent=2))
                        continue

                    # Get match IDs
                    match_ids = self.api.get_match_ids(puuid, count=matches_per_player)
                    if not match_ids:
                        print(f"No matches found for summoner ID {summoner_id}")
                        continue
                        
                    print(f"Found {len(match_ids)} matches for summoner ID {summoner_id}")

                    # Get and save match details
                    for match_id in match_ids:
                        match_file = os.path.join(self.raw_dir, f"{match_id}.json")
                        if not os.path.exists(match_file):
                            try:
                                match_data = self.api.get_match_details(match_id)
                                self._save_match(match_data, match_id)
                                matches_collected += 1
                                print(f"Collected match {match_id} ({matches_collected} total)")
                            except Exception as e:
                                print(f"Error collecting match {match_id}: {str(e)}")
                                traceback.print_exc()
                                continue

                except Exception as e:
                    print(f"Error processing player {summoner_id if summoner_id else idx}: {str(e)}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error in collect_challenger_matches: {str(e)}")
            traceback.print_exc()

    def process_matches(self) -> pd.DataFrame:
        """Process collected match data into a DataFrame."""
        matches_data = []
        
        # Read all JSON files in raw directory
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.raw_dir, filename), 'r') as f:
                        match_data = json.load(f)
                    
                    # Extract relevant features for each player
                    for participant in match_data['info']['participants']:
                        try:
                            player_data = {
                                'match_id': match_data['metadata']['match_id'],
                                'placement': participant['placement'],
                                'level': participant['level'],
                                'players_eliminated': participant['players_eliminated'],
                                'time_eliminated': participant['time_eliminated'],
                                'total_damage_to_players': participant['total_damage_to_players'],
                                'augments': participant.get('augments', []),  # Use get() with default empty list
                                'traits': [trait['name'] for trait in participant['traits'] if trait['tier_current'] > 0],
                                'units': [{'character_id': unit['character_id'], 
                                         'tier': unit['tier'],
                                         'items': unit.get('items', [])} for unit in participant['units']]  # Use get() for items
                            }
                            matches_data.append(player_data)
                        except KeyError as e:
                            print(f"Error processing participant in match {filename}: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Error processing match file {filename}: {str(e)}")
                    continue
        
        return pd.DataFrame(matches_data)

if __name__ == "__main__":
    collector = DataCollector()
    
    # Collect new matches
    print("Starting data collection...")
    collector.collect_challenger_matches(matches_per_player=5, max_players=30)
    
    # Process collected matches
    print("\nProcessing collected matches...")
    df = collector.process_matches()
    print(f"Processed {len(df)} player records from matches")
    
    # Save processed data
    output_file = os.path.join(collector.data_dir, f"processed_matches_{datetime.now().strftime('%Y%m%d')}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
