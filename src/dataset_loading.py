import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# "Fraud" is a relict from a previous version of the code. However, naming it this way does not affect functionality.
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(csv_path: str, batch_size: int = 512, test_size: float = 0.2, random_state: int = 42):
    """
    Loads, performs stratified split, scales 'Amount' and 'Time' strictly on 
    the training split to prevent data leakage, and returns DataLoaders with positive class weight.
    """
    df = pd.read_csv(csv_path)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratified split BEFORE feature scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler strictly on training data, then transform both sets
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

    # Create PyTorch datasets
    train_ds = FraudDataset(X_train.values, y_train.values)
    test_ds = FraudDataset(X_test.values, y_test.values)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute imbalance weight ratio (Negative_Count / Positive_Count)
    num_pos = float(y_train.sum())
    num_neg = float(len(y_train) - num_pos)
    pos_weight = num_neg / num_pos

    input_dim = X_train.shape[1]
    return train_loader, test_loader, pos_weight, input_dim