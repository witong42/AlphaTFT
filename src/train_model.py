import torch
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.tft_model import TFTAgent
from src.environment.tft_env import TFTEnvironment
from src.training.train import TFTTrainer

def main():
    # Define dimensions
    # State dim calculation:
    # 5 (scalar features) + 
    # 5 * 61 (shop units) + 
    # 9 * 61 (bench units) + 
    # 28 * 61 (board units) + 
    # 30 (items)
    STATE_DIM = 5 + (5 + 9 + 28) * 61 + 30
    
    # Action dim calculation:
    # 5 (buy actions) +
    # 9 (sell actions) +
    # (9 + 28) * (9 + 28) (move actions) +
    # 1 (reroll) +
    # 1 (level up)
    ACTION_DIM = 5 + 9 + (37 * 37) + 2
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = TFTAgent(STATE_DIM, ACTION_DIM, device=str(device))
    env = TFTEnvironment()
    trainer = TFTTrainer(agent, env)
    
    # Training parameters
    num_epochs = 100  # Start with fewer epochs for testing
    batch_size = 32
    data_path = "data/processed_matches_20250223.csv"
    
    print("Starting supervised learning training...")
    losses = trainer.train_supervised(data_path, num_epochs=num_epochs, batch_size=batch_size)
    
    # Save the trained model
    save_path = "models/tft_agent.pt"
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Plot training losses
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()
        print("Training loss plot saved as training_loss.png")
    except ImportError:
        print("matplotlib not installed, skipping loss plot")

if __name__ == "__main__":
    main()
