import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from opacus import PrivacyEngine

def train_dp_epoch(model, train_loader, optimizer, criterion, privacy_engine, target_delta, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(X_batch)

    # Compute current epsilon spend using RDP accountant
    epsilon = privacy_engine.get_epsilon(target_delta)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, epsilon


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds).ravel()
    all_targets = np.array(all_targets).ravel()

    # Calculate fraud utility metrics
    roc_auc = roc_auc_score(all_targets, all_preds)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc


def run_dp_experiment(
    train_loader,
    test_loader,
    model,
    optimizer,
    criterion,
    epochs: int = 10,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    target_delta: float = 1e-5,
    device: str = "cpu"
):
    """
    Wraps standard PyTorch objects into DP-equivalent objects via Opacus 
    and executes training over a specified number of epochs.
    """
    model = model.to(device)
    privacy_engine = PrivacyEngine()

    # Wrap model, optimizer, and data_loader for per-sample clipping and noise addition
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    history = {"epoch": [], "loss": [], "epsilon": [], "roc_auc": [], "pr_auc": []}

    for epoch in range(1, epochs + 1):
        loss, epsilon = train_dp_epoch(
            model, train_loader, optimizer, criterion, privacy_engine, target_delta, device
        )
        roc_auc, pr_auc = evaluate(model, test_loader, device)

        history["epoch"].append(epoch)
        history["loss"].append(loss)
        history["epsilon"].append(epsilon)
        history["roc_auc"].append(roc_auc)
        history["pr_auc"].append(pr_auc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | "
            f"ε: {epsilon:.2f} (δ={target_delta}) | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
        )

    return model, history