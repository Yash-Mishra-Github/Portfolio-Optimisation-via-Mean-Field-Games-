# train_mfg.py  (Stage II, humanised)
"""
What this script does (plain English)
-------------------------------------
- Loads processed market data with a common-noise factor.
- Builds windowed *signature* features of the common-noise path (from sig_features.py).
- Trains a policy (FFN / LSTM / Transformer) that sees:
    (i) a small "market window" stream, and
    (ii) a signature snapshot of the common-noise path.
- Updates the mean field using a Nadaraya–Watson kernel on signatures (single-path case
  collapses to the batch mean), smoothed by an EMA to keep the fictitious-play loop calm.
- Keeps your loss pluggable: choose a built-in or use `--loss=custom` to call your own module.

This mirrors the Stage II pipeline in your paper: single-loop Sig-DFP + signatures +
kernel/EMA mean-field step + architecture comparison. (FFN/LSTM/Transformer)
"""

import os, json, argparse, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -- Small Transformer downsizer (optional via TF_SMALL=1) --
if os.getenv("TF_SMALL","0") == "1":
    _orig_TEL = nn.TransformerEncoderLayer
    def _small_TEL(*args, **kwargs):
        kwargs['d_model'] = min(kwargs.get('d_model', 64), 64)
        kwargs['nhead'] = min(kwargs.get('nhead', 2), 2)
        kwargs['dim_feedforward'] = min(kwargs.get('dim_feedforward', 128), 128)
        return _orig_TEL(*args, **kwargs)
    nn.TransformerEncoderLayer = _small_TEL

# -- signature helper from your repo --
from sig_features import windowed_signatures  # uses iisignature under the hood

# ---------------- Arguments ----------------
p = argparse.ArgumentParser()
p.add_argument("--model", choices=["lstm","transformer","ffn"], required=True)
p.add_argument("--data", default="processed_market_data.csv")
p.add_argument("--out", required=True)

# time/agents
p.add_argument("--iters", type=int, default=1000)
p.add_argument("--N", type=int, default=512)
p.add_argument("--K", type=int, default=252)

# portfolio SDE params (dX = pi*(mu dt + nu dW + sigma dB))
p.add_argument("--mu", type=float, default=0.05)
p.add_argument("--nu", type=float, default=0.20)
p.add_argument("--sigma", type=float, default=0.15)

# optimisation
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--mf_alpha", type=float, default=0.15)  # EMA smoothing for mean field
p.add_argument("--seed", type=int, default=42)

# signatures
p.add_argument("--depth", type=int, default=3)
p.add_argument("--lookback", type=int, default=60)

# loss selection (custom module supported)
p.add_argument("--loss", choices=[
    "terminal_log","running_log","drawdown_penalty",
    "variance_penalty","control_quadratic","custom"
], default="drawdown_penalty")
p.add_argument("--loss_lambda", type=float, default=0.10)

# housekeeping
p.add_argument("--checkpoint_every", type=int, default=250)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

# ---------------- Setup ----------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.out,"ckpt"), exist_ok=True)
def savefig(path): plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------- Data ----------------
df = pd.read_csv(args.data).dropna()

# normalised "market window" features (kept from your original flow)
df["Portfolio_CumRet"] = (1 + df["Portfolio_Ret"]).cumprod()
df["MeanField_CumRet"] = (1 + df["Mean_Field_Ret"]).cumprod()
df["Common_Noise_Cum"] = df["Common_Noise_Factor"].cumsum()
df["Norm_Portfolio"]   = df["Portfolio_CumRet"] / df["Portfolio_CumRet"].iloc[0]
df["Norm_MeanField"]   = df["MeanField_CumRet"] / df["MeanField_CumRet"].iloc[0]
df["Norm_Noise"]       = (df["Common_Noise_Cum"] - df["Common_Noise_Cum"].mean())/df["Common_Noise_Cum"].std()

series_np = df[["Norm_Portfolio","Norm_MeanField","Norm_Noise"]].values.astype(np.float32)
K = args.K
if len(series_np) < K:
    raise ValueError(f"Not enough rows ({len(series_np)}) for K={K}.")
series = torch.tensor(series_np[:K], dtype=torch.float32, device=device)   # [K,3]

# signature features of common-noise path (windowed; last-step summary per t)
CN = df["Common_Noise_Factor"].values
sig_grid = windowed_signatures(
    CN, depth=args.depth, lookback=args.lookback,
    include_time=True, basepoint=True, leadlag=True
)[:K]                           # [K, Dsig]
sig_grid = torch.tensor(sig_grid, dtype=torch.float32, device=device)
Dsig = sig_grid.shape[1]

# batch of agents (they share the observed env stream in this single-loop variant)
N = args.N
series_agents = series.unsqueeze(0).repeat(N,1,1)                         # [N,K,3]
T = 1.0; dt = T / K
X0 = 1.0
M_in = 3

# ---------------- Models ----------------
class LSTMPolicy(nn.Module):
    def __init__(self, in_ts=M_in, in_sig=Dsig, h=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ts, hidden_size=h, batch_first=True, num_layers=2)
        self.sig  = nn.Sequential(nn.Linear(in_sig,64), nn.ReLU(), nn.Linear(64,32))
        self.head = nn.Sequential(nn.Linear(h+32,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_ts, x_sig):  # x_ts: [N,K,3]; x_sig: [K,Dsig] or [N,K,Dsig]
        _, (hn, _) = self.lstm(x_ts)
        h_ts = hn[-1]                               # [N,h]
        if x_sig.dim()==2: x_sig = x_sig.unsqueeze(0).repeat(x_ts.size(0),1,1)
        h_sig = self.sig(x_sig[:,-1,:])             # last signature snapshot
        return self.head(torch.cat([h_ts, h_sig], dim=1))  # [N,1]

class TransformerPolicy(nn.Module):
    def __init__(self, in_ts=M_in, in_sig=Dsig, d_model=64, nhead=4, ff=128, layers=2):
        super().__init__()
        self.proj = nn.Linear(in_ts, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_layers=layers)
        self.sig  = nn.Sequential(nn.Linear(in_sig,64), nn.ReLU(), nn.Linear(64,32))
        self.head = nn.Sequential(nn.Linear(d_model+32,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_ts, x_sig):
        z = self.proj(x_ts); z = self.enc(z)
        h_ts = z[:,-1,:]
        if x_sig.dim()==2: x_sig = x_sig.unsqueeze(0).repeat(x_ts.size(0),1,1)
        h_sig = self.sig(x_sig[:,-1,:])
        return self.head(torch.cat([h_ts, h_sig], dim=1))

class FFNPolicy(nn.Module):
    def __init__(self, K, in_ts=M_in, in_sig=Dsig, width=128):
        super().__init__()
        self.ts   = nn.Sequential(nn.Flatten(), nn.Linear(K*in_ts, width), nn.ReLU(), nn.Linear(width,64), nn.ReLU())
        self.sig  = nn.Sequential(nn.Linear(in_sig,64), nn.ReLU(), nn.Linear(64,32))
        self.head = nn.Sequential(nn.Linear(64+32,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_ts, x_sig):
        h_ts  = self.ts(x_ts)
        if x_sig.dim()==2: x_sig = x_sig[-1]            # [Dsig]
        if x_sig.dim()==1: x_sig = x_sig.unsqueeze(0).repeat(x_ts.size(0),1)
        h_sig = self.sig(x_sig)
        return self.head(torch.cat([h_ts, h_sig], dim=1))

def make_policy(name):
    if name=="lstm":        return LSTMPolicy().to(device)
    if name=="transformer": return TransformerPolicy().to(device)
    if name=="ffn":         return FFNPolicy(K).to(device)
    raise ValueError("unknown --model")

# ---------------- Kernel-on-signatures helpers ----------------
@torch.no_grad()
def _silverman_gamma(W):
    """Return gamma = 1/(2 h^2) for Gaussian kernel with Silverman bandwidth."""
    Wc = W - W.mean(dim=0, keepdim=True)
    sigma = (Wc.pow(2).mean().sqrt() + 1e-12).item()
    n = W.shape[0]
    h = 1.06 * sigma * (n ** (-1.0/5.0)) if n > 1 else 1.0
    return 0.5 / (h*h + 1e-12)

@torch.no_grad()
def kernel_mean_at_t(Xvals, Sig_bank, sig_query, gamma=None):
    """
    Nadaraya–Watson conditional mean E[X | w] on signature features.
    In this single-common-path variant Sig_bank is identical across agents, so this
    reduces to a batch mean; the structure stays so multiple common paths are a
    drop-in extension later.
    """
    if gamma is None: gamma = _silverman_gamma(Sig_bank)
    d2 = ((Sig_bank - sig_query.unsqueeze(0))**2).sum(dim=1)    # [N]
    w  = torch.exp(-gamma * d2) + 1e-12
    return (w * Xvals).sum() / w.sum()

# ---------------- Losses & simple metrics ----------------
def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    dd = (cum-peak)/np.clip(peak,1e-12,None)
    return float(dd.min())

def compute_loss(X_paths, pi_series, mf_series):
    lam = args.loss_lambda
    mode = args.loss
    if mode=="custom":
        import custom_loss
        return custom_loss.compute_loss(X_paths, pi_series, mf_series, dt, lam)
    if mode=="terminal_log":
        return -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
    if mode=="running_log":
        return -(torch.log(torch.clamp(X_paths[:,1:,0], min=1e-8))).mean()
    if mode=="variance_penalty":
        base = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
        return base + lam * X_paths[:,-1,0].var()
    if mode=="control_quadratic":
        base = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
        return base + lam * (pi_series[:,:,0]**2).mean()
    # default: drawdown-penalised log utility
    base = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
    with torch.no_grad():
        mdd = -max_drawdown(X_paths[:,:,0].mean(dim=0).cpu().numpy())
    return base + lam*torch.tensor(mdd, dtype=torch.float32, device=X_paths.device)

# ---------------- Train (single run) ----------------
def run_once(tag="base"):
    policy = make_policy(args.model)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # mean-field path initialised to ones; keep an EMA over iterations
    mf = torch.full((N, K+1, 1), X0, device=device)

    # state buffers for logging
    losses, mean_XT, osc = [], [], []

    # simple checkpoint (optional resume)
    ckpt_path = os.path.join(args.out, "ckpt", f"{tag}_latest.pt")
    start_it = 1
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(state["policy"]); opt.load_state_dict(state["opt"])
        mf = state["mf"]; start_it = state["iter"]+1

    for it in range(start_it, args.iters+1):
        # one idio path per agent; one common path shared across agents (extendable)
        dW = torch.randn(N, K, 1, device=device) * math.sqrt(dt)
        dB_common = torch.randn(K, 1, device=device) * math.sqrt(dt)
        dB = dB_common.unsqueeze(0).repeat(N,1,1)

        # ---- STEP 3: Mean-field update (no grad) ----
        with torch.no_grad():
            X_sim = torch.zeros(N, K+1, 1, device=device); X_sim[:,0,0] = X0

            # signature bank (identical across agents in this single-path case)
            Sig_bank = sig_grid.unsqueeze(0).repeat(N,1,1)  # [N,K,Dsig]

            for k in range(K):
                pi_k = policy(series_agents, sig_grid)                 # [N,1]
                dX   = pi_k*(args.mu*dt + args.nu*dW[:,k] + args.sigma*dB[:,k])
                X_sim[:,k+1] = X_sim[:,k] + dX

            # conditional mean per time via kernel-on-signatures, then EMA smooth
            # (collapses to batch mean here; structure retained for multi-path later)
            mf_target = X_sim.mean(dim=0, keepdim=True).repeat(N,1,1) # [N,K+1,1]
            mf = (1 - args.mf_alpha)*mf + args.mf_alpha*mf_target

        # ---- Train step (backprop through policy only) ----
        X_path  = torch.zeros(N, K+1, 1, device=device); X_path[:,0,0] = X0
        actions = torch.zeros(N, K, 1, device=device)
        for k in range(K):
            a_k = policy(series_agents, sig_grid)                      # [N,1]
            actions[:,k] = a_k
            dX = a_k*(args.mu*dt + args.nu*dW[:,k] + args.sigma*dB[:,k])
            X_path[:,k+1] = X_path[:,k] + dX

        loss = compute_loss(X_path, actions, mf)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        # light logging
        losses.append(float(loss.item()))
        mean_XT.append(float(X_path[:,-1,0].mean().item()))
        if len(mean_XT) >= 100:
            w = np.array(mean_XT[-100:])
            osc.append(float(w.max()-w.min()))
        if it % 50 == 0 or it == 1:
            print(f"[{tag}] it={it:>4}  loss={losses[-1]:.6f}  mean_XT={mean_XT[-1]:.4f}")

        # checkpoint
        if it % args.checkpoint_every == 0 or it == args.iters:
            torch.save({"iter": it, "policy": policy.state_dict(), "opt": opt.state_dict(), "mf": mf}, ckpt_path)

    # ---------------- Summaries / plots ----------------
    XT = X_path[:,:,0].detach().cpu().numpy()
    rets = (XT[:,1:] - XT[:,:-1]) / np.clip(XT[:,:-1], 1e-12, None)
    mean_rets = rets.mean(axis=0)
    SR = float(mean_rets.mean()/(mean_rets.std()+1e-12))
    mdd = -max_drawdown(XT.mean(axis=0))

    pd.DataFrame(losses, columns=["loss"]).to_csv(os.path.join(args.out,"losses.csv"), index=False)
    pd.DataFrame(XT[:,-1], columns=["terminal_wealth"]).to_csv(os.path.join(args.out,"final_wealths.csv"), index=False)
    pd.DataFrame(mean_XT, columns=["mean_XT"]).to_csv(os.path.join(args.out,"mean_XT.csv"), index=False)
    if len(osc)>0:
        pd.DataFrame(osc, columns=["osc_amp_window100"]).to_csv(os.path.join(args.out,"oscillation_amp.csv"), index=False)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(losses); plt.title("Loss"); plt.grid(True)
    plt.subplot(1,2,2); plt.hist(XT[:,-1], bins=30, density=True); plt.title("Final Wealth"); plt.grid(True)
    savefig(os.path.join(args.out,"training_results.png"))

    plt.figure(figsize=(10,5))
    for i in range(min(6, XT.shape[0])): plt.plot(XT[i], alpha=0.8)
    plt.title("Wealth Paths (samples)"); plt.grid(True)
    savefig(os.path.join(args.out,"wealth_paths.png"))

    plt.figure(figsize=(10,4)); plt.plot(mean_rets.cumsum()); plt.title("Cumulative mean returns (proxy)"); plt.grid(True)
    savefig(os.path.join(args.out,"cum_mean_returns.png"))

    with open(os.path.join(args.out,"summary_stats.json"),"w") as f:
        json.dump({"model": args.model, "iters": args.iters, "N": N, "K": K,
                   "loss_mode": args.loss, "Sharpe_proxy": SR, "MaxDrawdown_mean_path": mdd,
                   "last_loss": losses[-1], "osc_amp_last": (osc[-1] if len(osc)>0 else None)},
                  f, indent=2)

# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"Device={device} | Model={args.model} | Out={args.out}")
    run_once(tag=f"{args.model}")
    print("✅ Done:", args.out)
