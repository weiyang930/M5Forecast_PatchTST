import torch
from model import load_model_from_checkpoint

# Load model from config and weights
model = load_model_from_checkpoint(
    config_path='checkpoints/your_experiment/args.yaml',
    checkpoint_path='checkpoints/your_experiment/checkpoint.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create dummy input for testing [B, L, D]
x = torch.randn(1, model.seq_len, model.enc_in).to(model.device)

# Forward pass
with torch.no_grad():
    pred = model(x, itr=0)

print("Prediction shape:", pred.shape)
