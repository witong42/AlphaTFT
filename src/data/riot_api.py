import os
import requests
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

class RiotAPI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('RIOT_API_KEY')
        if not self.api_key:
            raise ValueError("RIOT_API_KEY not found in environment variables")

        self.base_url = "https://europe.api.riotgames.com/tft"
        self.region_url = "https://euw1.api.riotgames.com/tft"
        self.headers = {
            "X-Riot-Token": self.api_key
        }

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Handle rate limiting by waiting if necessary."""
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return True
        elif response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        return False

    def get_challenger_players(self) -> Dict[str, Any]:
        """Get list of challenger players."""
        url = f"{self.region_url}/league/v1/challenger"
        response = requests.get(url, headers=self.headers)

        if self._handle_rate_limit(response):
            return self.get_challenger_players()

        response.raise_for_status()
        return response.json()

    def get_match_ids(self, puuid: str, count: int = 20) -> List[str]:
        """Get match IDs for a player."""
        url = f"{self.base_url}/match/v1/matches/by-puuid/{puuid}/ids"
        params = {'count': count}
        response = requests.get(url, headers=self.headers, params=params)

        if self._handle_rate_limit(response):
            return self.get_match_ids(puuid, count)

        response.raise_for_status()
        return response.json()

    def get_match_details(self, match_id: str) -> Dict[str, Any]:
        """Get detailed match information."""
        url = f"{self.base_url}/match/v1/matches/{match_id}"
        response = requests.get(url, headers=self.headers)

        if self._handle_rate_limit(response):
            return self.get_match_details(match_id)

        response.raise_for_status()
        return response.json()

    def get_summoner_by_id(self, summoner_id: str) -> Dict[str, Any]:
        """Get summoner information by summoner ID."""
        url = f"{self.region_url}/summoner/v1/summoners/{summoner_id}"
        response = requests.get(url, headers=self.headers)

        if self._handle_rate_limit(response):
            return self.get_summoner_by_id(summoner_id)

        if response.status_code == 404:
            print(f"Summoner ID {summoner_id} not found")
            return {}

        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    # Example usage
    api = RiotAPI()

    # Get challenger players
    challenger_data = api.get_challenger_players()
    players = challenger_data.get('entries', [])
    print(f"Found {len(players)} challenger players")

    # Get matches for first challenger
    if players:
        first_player = players[0]
        summoner_id = first_player.get('summonerId')
        if summoner_id:
            summoner_info = api.get_summoner_by_id(summoner_id)
            puuid = summoner_info.get('puuid')

            if puuid:
                # Get recent matches
                matches = api.get_match_ids(puuid, count=5)
                print(f"Found {len(matches)} matches for summoner ID {summoner_id}")

                # Get details for first match
                if matches:
                    match_details = api.get_match_details(matches[0])
                    print(f"Match details retrieved for match {matches[0]}")
