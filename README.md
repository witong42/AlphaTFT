# AlphaTFT

An AI model that learns to play Teamfight Tactics (TFT) using supervised learning on data collected from high-ranked players via Riot's API.

## Project Overview
The project aims to:
1. Collect TFT match data from high-ranked players using Riot's API
2. Process and analyze the data to understand winning strategies
3. Train a supervised learning model to predict optimal actions
4. Evaluate the model's performance

## Setup
1. Install required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements.txt --break-system-packages
```

2. Set up your Riot API key:
- Get your API key from [Riot Developer Portal](https://developer.riotgames.com/)
- Create a `.env` file and add your API key:
```
RIOT_API_KEY=your-api-key-here
```

3. Run the data collection script:
```bash
python3.13 src/data/collect_data.py
head -n 5 data/processed_matches_20250223.csv
python3 src/train_model.py
PYTHONPATH=/Users/william-mbp/Documents/AlphaTFT python3 src/play_tft.py
```

## Project Structure
```
AlphaTFT/
├── data/               # Raw and processed game data
├── src/               # Source code
│   ├── data/          # Data collection and processing
│   ├── models/        # ML models
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for analysis
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## License
MIT License
