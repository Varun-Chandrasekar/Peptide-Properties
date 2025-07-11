"""
main.py

Runs training of the CNN-based peptide property predictor.
"""

from peptide_properties.preprocess import load_sequences, compute_targets, amino_acids, aa_to_int
from peptide_properties.model import CNN

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PeptideDataset(Dataset):
    def __init__(self, sequences, targets, max_length):
        self.sequences = sequences
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def one_hot_encode(self, seq):
        one_hot = np.zeros((self.max_length, len(amino_acids)), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_length]):
            if aa in aa_to_int:
                one_hot[i, aa_to_int[aa]] = 1
        return one_hot

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        one_hot = self.one_hot_encode(seq)
        return torch.tensor(one_hot), torch.tensor(target, dtype=torch.float32)

def train():
    sequences = load_sequences()
    targets = compute_targets(sequences)

    # Normalize targets
    targets = np.array(targets)
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    targets_scaled = (targets - mean) / std

    # Save mean and std
    np.save('target_mean.npy', mean)
    np.save('target_std.npy', std)

    max_length = max(len(seq) for seq in sequences)
    dataset = PeptideDataset(sequences, targets_scaled, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        epoch_loss = 0.0
        for x, y in loader:
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

if __name__ == "__main__":
    train()
