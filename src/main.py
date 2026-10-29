import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
from scipy.signal import savgol_coeffs
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Function
from torch.utils.data import DataLoader as PyDataLoader
from torch.utils.data import TensorDataset

from dataloader import ProjectDataLoader
from preprocessing import SpectralPreprocessor


def get_args_parser():
    parser = argparse.ArgumentParser("BISN LOBO evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw", "preprocessed"],
    )
    parser.add_argument(
        "--b_out",
        type=int,
        default=1,
        help="Physical batch label to hold out.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./trained_model_bisn",
    )
    return parser


class GradientReversalFunction(Function):
    """Identity in the forward pass and unit sign reversal in backpropagation."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class GRL(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)


class InformedPreprocessing(nn.Module):
    """Learnable SG-initialized convolution followed by instance normalization."""

    def __init__(
        self,
        n_features,
        window_size=61,
        poly_order=1,
        deriv_order=1,
    ):
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        if window_size > n_features:
            raise ValueError("window_size cannot exceed the number of features.")

        self.n_features = n_features
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=window_size,
            padding=window_size // 2,
            bias=False,
        )

        coefficients = savgol_coeffs(
            window_length=window_size,
            polyorder=poly_order,
            deriv=deriv_order,
        )
        sg_kernel = (
            torch.from_numpy(coefficients[::-1].copy())
            .float()
            .view(1, 1, -1)
        )
        self.conv.weight = nn.Parameter(sg_kernel, requires_grad=True)
        self.norm = nn.InstanceNorm1d(1, affine=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.norm(x)
        x = x.squeeze(1)

        if x.shape[1] > self.n_features:
            trim = (x.shape[1] - self.n_features) // 2
            x = x[:, trim : trim + self.n_features]

        return x


class BatchInvariantTabNet(nn.Module):
    def __init__(
        self,
        input_dim,
        n_species_classes,
        n_auxiliary_outputs,
        n_d=8,
        n_a=16,
        n_steps=1,
    ):
        super().__init__()

        self.preprocessing = InformedPreprocessing(input_dim)
        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=64,
            momentum=0.02,
            mask_type="sparsemax",
        )
        self.species_head = nn.Linear(n_d, n_species_classes)
        self.grl = GRL()

        # The historical attribute name is retained so that the state-dict keys
        # remain compatible with the trained BISN checkpoints. Its outputs are
        # unsupervised auxiliary slots; no batch labels enter the loss.
        self.batch_discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_auxiliary_outputs),
        )

    def to(self, device):
        super().to(device)
        self.tabnet.to(device)

        if hasattr(self.tabnet, "encoder") and hasattr(
            self.tabnet.encoder,
            "group_attention_matrix",
        ):
            self.tabnet.encoder.group_attention_matrix = (
                self.tabnet.encoder.group_attention_matrix.to(device)
            )

        return self

    def forward(self, x_raw):
        x_hat = self.preprocessing(x_raw)
        auxiliary_logits = self.batch_discriminator(self.grl(x_hat))
        latent_embedding, sparsity_loss = self.tabnet(x_hat)
        species_logits = self.species_head(latent_embedding)
        return species_logits, auxiliary_logits, sparsity_loss


def get_lambda_schedule(epoch, max_epochs, max_lambda=1.0):
    """Anneal lambda from zero toward max_lambda."""
    progress = epoch / max_epochs
    return max_lambda * (
        2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    )


def normalized_auxiliary_entropy(auxiliary_logits):
    probabilities = torch.softmax(auxiliary_logits, dim=1)
    entropy = -(
        probabilities * torch.log(probabilities + 1e-8)
    ).sum(dim=1).mean()
    return entropy / np.log(auxiliary_logits.shape[1])


def train_bisn(
    model,
    train_loader,
    X_valid,
    y_valid,
    model_params,
    device,
):
    """Train BISN and select the checkpoint using validation species accuracy."""
    criterion_species = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params["LR"])

    y_valid_tensor = torch.as_tensor(
        y_valid,
        dtype=torch.long,
        device=device,
    )

    best_val_acc = -np.inf
    best_epoch = None
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {
        "epoch": [],
        "lambda": [],
        "train_loss": [],
        "val_species_loss": [],
        "val_acc": [],
        "val_auxiliary_entropy_norm": [],
    }

    for epoch in range(model_params["MAX_EPOCHS"]):
        lambda_val = get_lambda_schedule(
            epoch,
            model_params["MAX_EPOCHS"],
            model_params["LAMBDA_MAX"],
        )
        model.train()
        epoch_losses = []

        for x_batch, y_species in train_loader:
            optimizer.zero_grad()

            species_logits, auxiliary_logits, sparsity_loss = model(x_batch)
            loss_species = criterion_species(species_logits, y_species)

            auxiliary_probabilities = torch.softmax(
                auxiliary_logits,
                dim=1,
            )
            loss_auxiliary = (
                auxiliary_probabilities
                * torch.log(auxiliary_probabilities + 1e-8)
            ).sum(dim=1).mean()

            # Lambda is applied exactly once, here in the joint objective.
            loss = (
                loss_species
                + lambda_val * loss_auxiliary
                + model_params["BETA"] * sparsity_loss
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_species_logits, val_auxiliary_logits, _ = model(X_valid)
            val_predictions = val_species_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_valid, val_predictions)
            val_species_loss = criterion_species(
                val_species_logits,
                y_valid_tensor,
            ).item()
            val_entropy = normalized_auxiliary_entropy(
                val_auxiliary_logits
            ).item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

        mean_train_loss = float(np.mean(epoch_losses))
        history["epoch"].append(epoch + 1)
        history["lambda"].append(lambda_val)
        history["train_loss"].append(mean_train_loss)
        history["val_species_loss"].append(val_species_loss)
        history["val_acc"].append(val_acc)
        history["val_auxiliary_entropy_norm"].append(val_entropy)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"train objective: {mean_train_loss:.4f} | "
            f"val species loss: {val_species_loss:.4f} | "
            f"val acc: {val_acc:.4f} | "
            f"aux entropy: {val_entropy:.4f} | "
            f"lambda: {lambda_val:.4f}"
        )

    if best_epoch is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")

    print(
        f"Best validation checkpoint: epoch {best_epoch}, "
        f"validation accuracy {best_val_acc:.4f}"
    )
    return best_model_wts, best_val_acc, best_epoch, history


def one_batch_leave_out_bisn(
    data_loader,
    holdout_batch,
    model_params,
    device,
    preprocessed=False,
    save_dir="./trained_model_bisn",
):
    dataframe = data_loader.get_joint()
    X_all = data_loader.X_nir
    y_species_all = dataframe["species"].values
    y_batch_all = dataframe["batch"].values

    unique_batches = np.unique(y_batch_all)
    if holdout_batch not in unique_batches:
        raise ValueError(
            f"Batch {holdout_batch} was not found. Available batches: "
            f"{unique_batches.tolist()}"
        )

    print("=" * 60)
    print(f"LOBO fold: physical batch {holdout_batch} held out")
    print("=" * 60)

    species_encoder = LabelEncoder()
    y_species_encoded = species_encoder.fit_transform(y_species_all)

    test_indices = np.where(y_batch_all == holdout_batch)[0]
    source_indices = np.where(y_batch_all != holdout_batch)[0]

    X_source = X_all[source_indices]
    y_source = y_species_encoded[source_indices]
    X_test = X_all[test_indices]
    y_test = y_species_encoded[test_indices]

    if preprocessed: # Apply the best preprocessing configuration for the held-out batch.
        configurations = pd.read_csv(
            "./outputs/best_preproc_configurations_nir.csv"
        )
        row = configurations.loc[
            configurations["left_out_batch"] == holdout_batch
        ]
        if row.empty:
            raise ValueError(
                f"No preprocessing configuration for batch {holdout_batch}."
            )
        row = row.iloc[0]
        preprocessor = SpectralPreprocessor(
            deriv_order=row["deriv_order"],
            sg_window=row["sg_window"],
            sg_polyorder=row["sg_polyorder"],
            use_simple_deriv=row["use_simple_deriv"],
            use_snv=row["use_snv"],
            use_msc=row["use_msc"],
            detrend_degree=row["detrend_degree"],
            center_cols=row["center_cols"],
        )
        preprocessor.fit(X_source)
        X_source = preprocessor.transform(X_source)
        X_test = preprocessor.transform(X_test)

    train_indices, valid_indices = train_test_split(
        np.arange(len(X_source)),
        test_size=0.20,
        random_state=10,
        stratify=y_source,
    )

    X_train_tensor = torch.as_tensor(
        X_source[train_indices],
        dtype=torch.float32,
        device=device,
    )
    y_train_tensor = torch.as_tensor(
        y_source[train_indices],
        dtype=torch.long,
        device=device,
    )
    X_valid_tensor = torch.as_tensor(
        X_source[valid_indices],
        dtype=torch.float32,
        device=device,
    )
    X_test_tensor = torch.as_tensor(
        X_test,
        dtype=torch.float32,
        device=device,
    )

    # Batch labels are deliberately absent from the training dataset and loss.
    training_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    training_loader = PyDataLoader(
        training_dataset,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
    )

    n_source_batches = len(np.unique(y_batch_all[source_indices]))
    model = BatchInvariantTabNet(
        input_dim=model_params["input_dim"],
        n_species_classes=model_params["n_species_classes"],
        n_auxiliary_outputs=n_source_batches,
        n_d=model_params["n_d"],
        n_a=model_params["n_a"],
        n_steps=model_params["n_steps"],
    ).to(device)

    print(
        f"Train: {len(train_indices)} | validation: {len(valid_indices)} | "
        f"held-out test: {len(test_indices)}"
    )
    best_weights, best_val_acc, best_epoch, history = train_bisn(
        model=model,
        train_loader=training_loader,
        X_valid=X_valid_tensor,
        y_valid=y_source[valid_indices],
        model_params=model_params,
        device=device,
    )

    # The held-out batch is evaluated once, after validation-only selection.
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_species_logits, test_auxiliary_logits, _ = model(X_test_tensor)
        test_predictions = test_species_logits.argmax(dim=1).cpu().numpy()
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions, average="weighted")
        test_auxiliary_entropy = normalized_auxiliary_entropy(
            test_auxiliary_logits
        ).item()

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        save_dir,
        f"BISN_LOBO_{holdout_batch}.pt",
    )
    history_path = os.path.join(
        save_dir,
        f"BISN_history_LOBO_{holdout_batch}.csv",
    )
    torch.save(best_weights, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)

    result = {
        "holdout_batch": holdout_batch,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_accuracy,
        "test_f1": test_f1,
        "test_auxiliary_entropy_norm": test_auxiliary_entropy,
        "checkpoint": checkpoint_path,
    }

    print("\n--- Result Summary ---")
    print(pd.Series(result))
    return result


def main(args):
    device = torch.device(args.device)
    data_loader = ProjectDataLoader(data_dir=args.data_dir, seed=42)
    data_loader.load_all()

    dataframe = data_loader.get_joint()
    model_params = {
        "input_dim": data_loader.X_nir.shape[1],
        "n_species_classes": dataframe["species"].nunique(),
        "n_d": 8,
        "n_a": 16,
        "n_steps": 1,
        "MAX_EPOCHS": 200,
        "BATCH_SIZE": 16,
        "LR": 1e-4,
        "LAMBDA_MAX": 1.0,
        "BETA": 1e-5,
    }

    one_batch_leave_out_bisn(
        data_loader=data_loader,
        holdout_batch=args.b_out,
        model_params=model_params,
        device=device,
        preprocessed=(args.mode == "preprocessed"),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
