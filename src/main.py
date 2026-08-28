import torch
import torch.optim as optim
from src.dataset_loading import prepare_data
from src.models import DPMLPClassifier
from src.loss_def import BinaryFocalLoss
from src.dp_engine import run_dp_experiment

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"
    BATCH_SIZE = 512
    EPOCHS = 10
    MAX_GRAD_NORM = 1.0       # Clipping threshold C
    NOISE_MULTIPLIER = 1.2     # Noise scale sigma
    TARGET_DELTA = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Data
    train_loader, test_loader, pos_weight, input_dim = prepare_data(
        DATA_PATH, batch_size=BATCH_SIZE
    )

    # Instantiate Model, Loss, and Optimizer
    model = DPMLPClassifier(input_dim=input_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BinaryFocalLoss(gamma=2.0, pos_weight=pos_weight)

    # Train with Differential Privacy
    model, history = run_dp_experiment(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=NOISE_MULTIPLIER,
        target_delta=TARGET_DELTA,
        device=DEVICE
    )