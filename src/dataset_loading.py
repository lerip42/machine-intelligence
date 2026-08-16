import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FraudDataset(Dataset):
    """PyTorch Dataset wrapper for Kaggle Credit Card Fraud data."""
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(csv_path: str, batch_size: int = 512, test_size: float = 0.2, random_state: int = 42):
    """
    Loads, scales 'Amount' and 'Time', performs stratified split, 
    and returns DataLoaders along with the positive class weight ratio.
    """
    df = pd.read_csv(csv_path)

    # Standardize 'Amount' and 'Time' features; V1-V28 are already PCA transformed
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_ds = FraudDataset(X_train, y_train)
    test_ds = FraudDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute imbalance weight ratio (Negative_Count / Positive_Count)
    num_pos = float(y_train.sum())
    num_neg = float(len(y_train) - num_pos)
    pos_weight = num_neg / num_pos

    input_dim = X_train.shape[1]
    return train_loader, test_loader, pos_weight, input_dim