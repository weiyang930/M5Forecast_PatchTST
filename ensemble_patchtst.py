import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from model import PatchTST


def generate_synthetic_csv(csv_path="series.csv", total_length=1000):
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 50, total_length)) + np.random.normal(0, 0.1, total_length)
    pd.DataFrame({"value": signal}).to_csv(csv_path, index=False)
    return csv_path


class TimeSeriesDataset(Dataset):
    def __init__(self, data, hist_len, pred_len):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.data = data
        self.samples = self._create_samples()

    def _create_samples(self):
        total = self.hist_len + self.pred_len
        return [
            (self.data[i:i+self.hist_len], self.data[i+self.hist_len:i+total])
            for i in range(len(self.data) - total)
        ]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


def load_patchtst_model(ckpt_path, config_path, device):
    with open(config_path, 'r') as f:
        args = Namespace(**yaml.safe_load(f))
    model = PatchTST(args, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def predict_with_checkpoints(checkpoint_paths, config_path, dataloader, device):
    predictions = []
    for path in checkpoint_paths:
        model = load_patchtst_model(path, config_path, device)
        preds = [model(x.to(device), itr=0).detach().cpu().numpy() for x, _ in dataloader]
        predictions.append(np.concatenate(preds, axis=0).squeeze(-1))  # [B, pred_len]
    return np.stack(predictions, axis=1)  # [B, N, pred_len]


def fit_linear_regression(predictions, targets):
    B, N, P = predictions.shape
    X = predictions.transpose(0,2,1).reshape(-1, N)  # [B*P, N]
    y = targets.reshape(-1)  # [B*P]
    model = LinearRegression().fit(X, y)
    weights = model.coef_  # [N]
    final_preds = np.einsum("bnp,n->bp", predictions, weights)
    return weights, final_preds


def save_results(weights, final_preds, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "weights.npy"), weights)
    np.save(os.path.join(out_dir, "predictions.npy"), final_preds)
    print("Learned weights:", weights)
    print(f"Saved weights and predictions to `{out_dir}`")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--hist_len', type=int, default=366, help='Length of input history window')
    parser.add_argument('--pred_len', type=int, default=28, help='Length of prediction horizon')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_checkpoints', type=int, default=3, help='Number of checkpoints to ensemble. The current maximum is 220.')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_patchtst/', help='Checkpoint directory')
    parser.add_argument('--config_path', type=str, default='./checkpoints_patchtst/args.yaml', help='Path to model config yaml')
    parser.add_argument('--csv_path', type=str, default='series.csv', help='Path to input time series csv')
    parser.add_argument('--out_dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    if not os.path.exists(args.csv_path):
        generate_synthetic_csv(args.csv_path)
    data = pd.read_csv(args.csv_path)["value"].values
    dataset = TimeSeriesDataset(data, args.hist_len, args.pred_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    targets = np.concatenate([y.numpy().squeeze(-1) for _, y in dataloader], axis=0)

    # Checkpoint ensemble
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "*.pth"), recursive=True))[:args.num_checkpoints]
    print(f"Using {len(ckpt_paths)} checkpoint(s):")

    predictions = predict_with_checkpoints(ckpt_paths, args.config_path, dataloader, device)
    weights, final_preds = fit_linear_regression(predictions, targets)
    save_results(weights, final_preds, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
