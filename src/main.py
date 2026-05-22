import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as PyDataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from dataloader import ProjectDataLoader
from preprocessing import SpectralPreprocessor
from model import BISN
import argparse


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("BISN LOBO Evaluation", add_help=False)
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "preprocessed"],
                        help="Input data: raw spectra or externally preprocessed spectra.")
    parser.add_argument("--b_out", type=int, default=1,
                        help="Index of the batch to leave out (0-based).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device (e.g. 'cuda:0' or 'cpu').")
    parser.add_argument("--save_dir", type=str, default="./trained_model",
                        help="Directory for saved model weights and training history.")
    return parser




# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(model: BISN,
          train_loader: PyDataLoader,
          X_valid: torch.Tensor,
          y_spec_valid: np.ndarray,
          X_train: torch.Tensor,
          y_batch_train: torch.Tensor,
          model_params: dict,
          device: torch.device,
          n_source_domains: int,
          history: dict,
          alpha: float = 1) -> tuple:
    """
    Train BISN with a single Adam optimiser.

    Joint objective (Eq. 1 in the paper):
        L = L_y  +  lambda(e) * L_b  +  beta * L_s

    where:
        L_y  = species cross-entropy                    [minimise]
        L_b  = negative Shannon entropy of batch probs  [minimise = maximise entropy]
        L_s  = TabNet sparsity regularisation            [minimise]
        beta = sparsity weight (model_params['BETA'])
        lambda(e) = annealed adversarial weight

    Model selection:
        composite_score = alpha * val_species_acc + (1 - alpha) * normalised_batch_entropy
        alpha=1 => select best model based on validation species accuracy alone
        alpha=0 => select best model based on batch entropy alone
        0 < alpha < 1 => weighted composite score for model selection
        

    Returns
    -------
    best_model_wts : OrderedDict
    best_composite_score : float
    best_epoch_stats : dict
    """

    criterion_species = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params["LR"])

    best_composite_score = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_stats = {}

    beta = model_params.get("BETA", 1e-3)

    print(f"  Training | alpha={alpha:.2f} | beta={beta} | "
          f"random batch acc = {1.0 / n_source_domains:.2f}")

    for epoch in range(model_params["MAX_EPOCHS"]):
        lambda_val = get_lambda(epoch, model_params["MAX_EPOCHS"],
                                model_params["LAMBDA_MAX"])
        model.set_grl_lambda(lambda_val)
        model.train()

        epoch_losses = []

        for x_batch, y_s, _ in train_loader:
            optimizer.zero_grad()

            species_logits, batch_logits, sparsity_loss = model(x_batch)

            # L_y: species cross-entropy
            loss_species = criterion_species(species_logits, y_s)

            # L_b: negative Shannon entropy — minimising this maximises entropy,
            #      pushing batch predictions toward a uniform distribution
            probs_batch = torch.softmax(batch_logits, dim=1)
            loss_batch = (probs_batch * torch.log(probs_batch + 1e-15)).sum(1).mean()

            # Joint objective
            loss = loss_species + lambda_val * loss_batch + beta * sparsity_loss
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            # Species accuracy on held-in validation set
            logits_val, _, _ = model(X_valid)
            preds_val = logits_val.argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_spec_valid, preds_val)

            # Validation species loss
            y_val_t = torch.tensor(y_spec_valid, dtype=torch.long, device=device)
            val_loss = criterion_species(logits_val, y_val_t).item()

            # Normalised batch entropy on training set (diagnostic + selection)
            _, logits_batch_train, _ = model(X_train)
            preds_batch = logits_batch_train.argmax(1).cpu().numpy()
            batch_acc = accuracy_score(y_batch_train.cpu().numpy(), preds_batch)

            probs_train_batch = torch.softmax(logits_batch_train, dim=1)
            entropy = -(probs_train_batch * torch.log(probs_train_batch + 1e-15)
                        ).sum(1).mean().item()
            norm_entropy = entropy / np.log(n_source_domains)  # scaled to [0, 1]

        # Model selection: composite score (no test data involved)
        composite_score = alpha * val_acc + (1.0 - alpha) * norm_entropy

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_stats = {
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "batch_acc": batch_acc,
                "norm_entropy": norm_entropy,
                "composite_score": composite_score,
            }

        print(f"  Ep {epoch+1:4d} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Batch Acc: {batch_acc:.3f} | "
              f"Entropy: {norm_entropy:.3f} | "
              f"Score: {composite_score:.3f} | "
              f"lambda: {lambda_val:.3f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        history["epoch"].append(epoch)
        history["val_acc"].append(val_acc)
        history["batch_acc"].append(batch_acc)
        history["norm_entropy"].append(norm_entropy)
        history["composite_score"].append(composite_score)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

    print(f"\n  Best model — epoch {best_epoch_stats['epoch']} | "
          f"Val Acc: {best_epoch_stats['val_acc']:.3f} | "
          f"Entropy: {best_epoch_stats['norm_entropy']:.3f} | "
          f"Score: {best_epoch_stats['composite_score']:.3f}")

    return best_model_wts, best_composite_score, best_epoch_stats


# =============================================================================
# LEAVE-ONE-BATCH-OUT EVALUATION
# =============================================================================

def leave_one_batch_out(dl, B_holdout: int, model_params: dict,
                        device: torch.device,
                        preprocessed: bool = False,
                        save_dir: str = "./trained_model") -> dict:
    """
    LOBO protocol:
      1. Hold out batch B_holdout as the strictly independent test set.
      2. Split remaining batches into train (80%) and internal validation (20%).
      3. Train on train split; select best model on composite validation score.
      4. Evaluate the selected model once on the held-out test batch.

    Test data is never accessed during training or model selection.
    """

    df = dl.get_joint()
    X_all = dl.X_nir
    y_species_all = df["species"].values
    y_batch_all = df["batch"].values

    unique_batches = np.unique(y_batch_all)
    if isinstance(B_holdout, int) and B_holdout < len(unique_batches):
        holdout_label = (unique_batches[B_holdout]
                         if B_holdout not in unique_batches else B_holdout)
    else:
        holdout_label = B_holdout

    print(f"\n{'='*60}")
    print(f"LOBO Fold: Held-out batch = '{holdout_label}'")
    print(f"{'='*60}")

    # Encode species labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_species_all)

    test_idx = np.where(y_batch_all == holdout_label)[0]
    src_idx = np.where(y_batch_all != holdout_label)[0]

    X_source, y_source = X_all[src_idx], y_encoded[src_idx]
    X_test, y_test = X_all[test_idx], y_encoded[test_idx]

    # Optional: replace raw spectra with externally optimised preprocessing
    if preprocessed:
        print("  Applying external preprocessing configuration...")
        preproc_cfg = pd.read_csv("./outputs/best_preproc_configurations_nir.csv")
        row = preproc_cfg.iloc[B_holdout - 1]
        sp = SpectralPreprocessor(
            deriv_order=row["deriv_order"],
            sg_window=row["sg_window"],
            sg_polyorder=row["sg_polyorder"],
            use_simple_deriv=row["use_simple_deriv"],
            use_snv=row["use_snv"],
            use_msc=row["use_msc"],
            detrend_degree=row["detrend_degree"],
            center_cols=row["center_cols"]
        )
        sp.fit(X_source)
        X_source = sp.transform(X_source)
        X_test = sp.transform(X_test)

    # Train / internal validation split — stratified by species
    train_idx, valid_idx = train_test_split(
        np.arange(len(X_source)),
        test_size=0.20,
        random_state=42,
        stratify=y_source
    )

    # Remap batch labels to contiguous integers for the discriminator
    source_batches = y_batch_all[src_idx]
    unique_src_batches = np.unique(source_batches)
    batch_map = {lbl: i for i, lbl in enumerate(unique_src_batches)}
    n_source_domains = len(unique_src_batches)

    print(f"  Train: {len(train_idx)} | Val: {len(valid_idx)} | "
          f"Test: {len(test_idx)} | "
          f"Source domains: {n_source_domains} | "
          f"Random-chance batch acc: {1.0 / n_source_domains:.2f}")

    # Build tensors
    X_train_t = torch.tensor(X_source[train_idx], dtype=torch.float32, device=device)
    y_spec_train_t = torch.tensor(y_source[train_idx], dtype=torch.long, device=device)
    y_batch_train_t = torch.tensor(
        [batch_map[b] for b in source_batches[train_idx]],
        dtype=torch.long, device=device
    )
    X_valid_t = torch.tensor(X_source[valid_idx], dtype=torch.float32, device=device)
    y_spec_valid = y_source[valid_idx]          # numpy for sklearn

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_spec_test = y_test                        # numpy for sklearn

    train_ds = TensorDataset(X_train_t, y_spec_train_t, y_batch_train_t)
    train_loader = PyDataLoader(
        train_ds,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        drop_last=True
    )

    # Instantiate model
    model = BISN(
        input_dim=model_params["input_dim"],
        n_species_classes=model_params["n_species_classes"],
        n_batch_classes=n_source_domains,
        n_d=model_params["n_d"],
        n_a=model_params["n_a"],
        n_steps=model_params["n_steps"]
    ).to(device)

    # Training
    history = {
        "epoch": [], "val_acc": [], "batch_acc": [],
        "norm_entropy": [], "composite_score": [],
        "train_loss": [], "val_loss": []
    }

    best_model_wts, best_score, best_epoch_stats = train(
        model=model,
        train_loader=train_loader,
        X_valid=X_valid_t,
        y_spec_valid=y_spec_valid,
        X_train=X_train_t,
        y_batch_train=y_batch_train_t,
        model_params=model_params,
        device=device,
        n_source_domains=n_source_domains,
        history=history,
        alpha=model_params.get("ALPHA", 1)
    )

    # Final evaluation — test data accessed exactly once
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        logits_test, batch_logits_test, _ = model(X_test_t)
        preds_test = logits_test.argmax(1).cpu().numpy()
        test_acc = accuracy_score(y_spec_test, preds_test)
        test_f1 = f1_score(y_spec_test, preds_test, average="weighted")

        # Normalised batch entropy on test set (OOD diagnostic)
        probs_test = torch.softmax(batch_logits_test, dim=1)
        entropy_test = -(probs_test * torch.log(probs_test + 1e-15)).sum(1).mean().item()
        norm_entropy_test = entropy_test / np.log(n_source_domains)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT | Held-out batch: '{holdout_label}'")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1 (wtd): {test_f1:.4f}")
    print(f"  Test Batch Entropy (norm.): {norm_entropy_test:.4f}  "
          f"[1.0 = uniform = batch-invariant]")
    print(f"  Best epoch: {best_epoch_stats['epoch']} | "
          f"Val Acc: {best_epoch_stats['val_acc']:.4f}")
    print(f"{'='*60}\n")

    # Persist model weights and training history
    torch.save(best_model_wts, f"{save_dir}/BISN_LOBO_{holdout_label}.pt")
    pd.DataFrame(history).to_csv(
        f"{save_dir}/BISN_history_LOBO_{holdout_label}.csv", index=False
    )

    return {
        "holdout_batch": holdout_label,
        "best_val_score": best_score,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_batch_entropy_norm": norm_entropy_test,
        "best_epoch": best_epoch_stats["epoch"],
    }

# =============================================================================
# MAIN
# =============================================================================

def main(args):
    dl = ProjectDataLoader(data_dir="./data", seed=42)
    dl.load_all()

    X_raw = dl.X_nir
    w = dl.wavelengths
    df = dl.get_joint()

    print(f"Data loaded | spectra: {X_raw.shape} | "
          f"wavelengths: {w[0]:.1f} ... {w[-1]:.1f} nm")

    input_dim = X_raw.shape[1]
    n_species_classes = df["species"].nunique()
    n_unique_batches = df["batch"].nunique()

    model_params = {
        "input_dim":         input_dim,
        "n_species_classes": n_species_classes,
        "n_batch_classes":   n_unique_batches,
        # Architecture (match paper values)
        "n_d":               8,
        "n_a":               16,
        "n_steps":           1,
        # Training
        "MAX_EPOCHS":        200,
        "BATCH_SIZE":        16,
        "LR":                1e-3,
        "LAMBDA_MAX":        1.0,
        "BETA":              1e-3,   # sparsity loss weight (beta in Eq. 1)
        "ALPHA":             1,    # composite score: alpha * val_acc + (1-alpha) * entropy, alpha=1 => select best model based on validation species accuracy alone
    }

    result = leave_one_batch_out(
        dl=dl,
        B_holdout=args.b_out,
        model_params=model_params,
        device=torch.device(args.device),
        preprocessed=(args.mode == "preprocessed"),
        save_dir=args.save_dir
    )

    print("\n--- Result Summary ---")
    print(pd.Series(result))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
    # Usage: python bisn.py --mode raw --device cuda:0 --b_out 1
