import torch
import numpy as np
from warnings import filterwarnings
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from opacus import PrivacyEngine

# Suppress Opacus Secure RNG warning during experimentation
filterwarnings("ignore", message=".*Secure RNG turned off.*")


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluates model performance across binary or multi-class datasets."""
    model.eval()
    all_targets = []
    all_logits = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_logits.append(logits.cpu())
        all_targets.append(y_batch.cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_targets, dim=0).numpy()

    # Multi-Class Evaluation (e.g., Fashion-MNIST with 10 classes)
    if logits.ndim > 1 and logits.shape[1] > 1:
        y_probs = torch.softmax(logits, dim=1).numpy()
        roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        pr_auc = 0.0  # PR-AUC is defined for binary metrics

    # Binary Evaluation (e.g., Credit Card Fraud)
    else:
        y_probs = torch.sigmoid(logits).squeeze().numpy()
        roc_auc = roc_auc_score(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)

    return roc_auc, pr_auc


def train_dp_epoch(model, loader, optimizer, criterion, privacy_engine, target_delta, device):
    """Trains the model for one epoch under DP-SGD."""
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        outputs = model(X_batch)
        
        # Ensure tensor shapes match binary targets
        if outputs.ndim > 1 and outputs.shape[1] == 1 and y_batch.ndim == 1:
            outputs = outputs.squeeze(1)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(X_batch)

    epoch_loss = running_loss / len(loader.dataset)
    epsilon = privacy_engine.get_epsilon(target_delta)
    return epoch_loss, epsilon


def run_dp_experiment(
    train_loader, 
    test_loader, 
    model, 
    optimizer, 
    criterion, 
    epochs, 
    max_grad_norm, 
    noise_multiplier, 
    target_delta=1e-5, 
    device="cpu"
):
    """Wraps model in Opacus PrivacyEngine and runs full DP-SGD training loop."""
    privacy_engine = PrivacyEngine()
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
            f"ε: {epsilon:.2f} (δ={target_delta}) | ROC-AUC: {roc_auc:.4f}"
        )

    return model, history