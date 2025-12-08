import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data: 1D Gaussian mixture
# -----------------------------

def sample_mixture(n_samples: int) -> torch.Tensor:
    """
    Sample from a 1D mixture of 3 Gaussians with means -4, 0, 4 and unit variance.
    Returns a tensor of shape (n_samples, 1).
    """
    means = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    std = 1.0

    # Choose components uniformly
    comps = np.random.choice(3, size=n_samples)
    data = np.random.randn(n_samples).astype(np.float32) * std + means[comps]

    return torch.from_numpy(data).unsqueeze(-1)  # (N, 1)


# -----------------------------
# 2. Time embedding + MLP
# -----------------------------

class Denoiser1D(nn.Module):
    """
    Simple MLP denoiser epsilon_theta(x_t, t).
    Input: x_t (batch, 1), discrete timestep t (batch,)
    Output: epsilon prediction (batch, 1)
    """

    def __init__(self, num_timesteps: int, time_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Discrete embedding for timesteps 0..T-1
        self.time_embed = nn.Embedding(num_timesteps, time_dim)

        self.net = nn.Sequential(
            nn.Linear(1 + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 1)
        t:   (B,) int64, each in [0, T-1]
        """
        # Embed time and concatenate with x_t
        t_emb = self.time_embed(t)  # (B, time_dim)
        x_in = torch.cat([x_t, t_emb], dim=-1)
        return self.net(x_in)


# -----------------------------
# 3. Main DDPM logic
# -----------------------------

def main():
    # Hyperparameters
    T = 1000               # number of diffusion steps
    BATCH_SIZE = 512
    NUM_STEPS = 10000      # training iterations
    LR = 1e-3
    PRINT_EVERY = 500

    RESULTS_DIR = os.path.join("results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Beta schedule (linear)
    beta_start = 1e-4
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)  # (T,)

    # Posterior variance (beta_tilde) for sampling
    beta_tilde = torch.zeros_like(beta)
    beta_tilde[0] = beta[0]  # not actually used for t=0 step
    beta_tilde[1:] = beta[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])

    model = Denoiser1D(num_timesteps=T).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    # --------- Helper functions using closure over alpha, alpha_bar, beta, beta_tilde ---------

    def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion (q) sample: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        x0:   (B, 1)
        t:    (B,)
        noise:(B, 1)
        """
        # Gather alpha_bar_t for each sample in the batch
        alpha_bar_t = alpha_bar[t].unsqueeze(-1)  # (B, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    @torch.no_grad()
    def p_sample_loop(n_samples: int) -> torch.Tensor:
        """
        Run the reverse diffusion process to sample from p_theta(x_0).
        Returns (n_samples, 1) on CPU.
        """
        model.eval()
        x_t = torch.randn(n_samples, 1, device=device)

        for t_step in reversed(range(T)):
            t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)

            # Predict noise epsilon
            eps_theta = model(x_t, t)  # (B, 1)

            alpha_t = alpha[t].unsqueeze(-1)          # (B, 1)
            beta_t = beta[t].unsqueeze(-1)            # (B, 1)
            alpha_bar_t = alpha_bar[t].unsqueeze(-1)  # (B, 1)
            beta_tilde_t = beta_tilde[t].unsqueeze(-1)

            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            # Equation for mean of p_theta(x_{t-1} | x_t)
            # mu_theta = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
            mu_theta = (
                1.0 / torch.sqrt(alpha_t)
                * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta)
            )

            if t_step > 0:
                # Add noise
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_tilde_t)
                x_t = mu_theta + sigma_t * noise
            else:
                # Last step: no noise
                x_t = mu_theta

        return x_t.cpu()

    def plot_loss_curve(losses):
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.xlabel("Training step")
        plt.ylabel("MSE loss")
        plt.title("DDPM training loss on 1D Gaussian mixture")
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "example1_loss_curve.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved loss curve to {out_path}")

    def plot_histograms(true_samples, model_samples):
        plt.figure(figsize=(6, 4))
        plt.hist(
            true_samples,
            bins=100,
            density=True,
            alpha=0.5,
            label="True data",
        )
        plt.hist(
            model_samples,
            bins=100,
            density=True,
            alpha=0.5,
            label="DDPM samples",
        )
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("1D Gaussian Mixture: True vs. DDPM model")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "example1_hist_data_vs_model.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved histogram comparison to {out_path}")

    # -----------------------------
    # 4. Training loop
    # -----------------------------

    losses = []
    model.train()

    for step in range(1, NUM_STEPS + 1):
        # Sample x0 from mixture
        x0 = sample_mixture(BATCH_SIZE).to(device)

        # Sample t ~ Uniform{0, ..., T-1}
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

        # Sample noise epsilon
        noise = torch.randn_like(x0)

        # Get x_t via the forward process
        x_t = q_sample(x0, t, noise)

        # Predict noise
        eps_pred = model(x_t, t)

        # DDPM simple loss: MSE between true noise and predicted noise
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % PRINT_EVERY == 0:
            print(f"Step {step}/{NUM_STEPS}, loss = {loss.item():.6f}")

    # -----------------------------
    # 5. Evaluation & plots
    # -----------------------------

    # Save training loss curve
    plot_loss_curve(losses)

    # Compare true vs. model samples (histograms)
    n_eval_samples = 50000
    true_samples = sample_mixture(n_eval_samples).numpy().reshape(-1)
    model_samples = p_sample_loop(n_eval_samples).numpy().reshape(-1)

    plot_histograms(true_samples, model_samples)


if __name__ == "__main__":
    main()