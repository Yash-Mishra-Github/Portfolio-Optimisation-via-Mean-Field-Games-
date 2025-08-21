import os, json, argparse, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# ---------------- arguments ----------------
P = argparse.ArgumentParser()
P.add_argument("--model", choices=["ffn","lstm","transformer"], required=True)
P.add_argument("--data", default="processed_market_data.csv")
P.add_argument("--out", required=True)
P.add_argument("--iters", type=int, default=1000)
P.add_argument("--N", type=int, default=512)     # number of agents
P.add_argument("--K", type=int, default=252)     # steps per episode
P.add_argument("--mu", type=float, default=0.05)
P.add_argument("--nu", type=float, default=0.20)
P.add_argument("--sigma", type=float, default=0.15)    # common noise scale
P.add_argument("--lr", type=float, default=5e-4)
P.add_argument("--mf_alpha", type=float, default=0.15) # EMA factor for mean field
P.add_argument("--seed", type=int, default=42)
P.add_argument("--depth", type=int, default=3)         # signature depth
P.add_argument("--lookback", type=int, default=60)     # signature window length
P.add_argument("--loss", choices=[
    "terminal_log","variance_penalty","control_quadratic","drawdown_penalty","custom"
], default="drawdown_penalty")
P.add_argument("--checkpoint_every", type=int, default=250)
args = P.parse_args()

# ---------------- setup ----------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.out,"ckpt"), exist_ok=True)
def savefig(path): plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------- data & signatures ----------------
df = pd.read_csv(args.data).dropna()
# normalised “market window” to keep the model grounded (same idea as train_quick)  # 
df["Portfolio_CumRet"] = (1 + df["Portfolio_Ret"]).cumprod()
df["MeanField_CumRet"] = (1 + df["Mean_Field_Ret"]).cumprod()
df["Common_Noise_Cum"] = df["Common_Noise_Factor"].cumsum()
df["Norm_Portfolio"]   = df["Portfolio_CumRet"] / df["Portfolio_CumRet"].iloc[0]
df["Norm_MeanField"]   = df["MeanField_CumRet"] / df["MeanField_CumRet"].iloc[0]
df["Norm_Noise"]       = (df["Common_Noise_Cum"] - df["Common_Noise_Cum"].mean())/df["Common_Noise_Cum"].std()

series_np = df[["Norm_Portfolio","Norm_MeanField","Norm_Noise"]].values.astype(np.float32)
if len(series_np) < args.K:
    raise ValueError(f"Need at least {args.K} rows in data; found {len(series_np)}.")
series = torch.tensor(series_np[:args.K], dtype=torch.float32, device=device)  # [K, 3]

# Signature features of the common-noise path (windowed, last-step summary per t)
CN = df["Common_Noise_Factor"].values
sig_grid = windowed_signatures(
    CN, depth=args.depth, lookback=args.lookback,
    include_time=True, basepoint=True, leadlag=True
)[:args.K]  # [K, Dsig]
sig_grid = torch.tensor(sig_grid, dtype=torch.float32, device=device)
Dsig = sig_grid.shape[1]

# Replicate the observed window for N agents (Sig-DFP’s single-loop spirit)  # 
N, K = args.N, args.K
series_agents = series.unsqueeze(0).repeat(N,1,1)  # [N, K, 3]
dt = 1.0 / K
X0 = 1.0

# ---------------- models ----------------
M_in = 3  # features in the small “market window” stream

class FFNPolicy(nn.Module):
    def __init__(self, K, in_ts=M_in, in_sig=Dsig, width=128):
        super().__init__()
        self.ts = nn.Sequential(nn.Flatten(), nn.Linear(K*in_ts, width), nn.ReLU(), nn.Linear(width,64), nn.ReLU())
        self.sig = nn.Sequential(nn.Linear(in_sig,64), nn.ReLU(), nn.Linear(64,32))
        self.head = nn.Sequential(nn.Linear(64+32,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_ts, x_sig):                  # x_ts: [N,K,3]; x_sig: [K,Dsig] or [N,K,Dsig]
        h_ts = self.ts(x_ts)
        if x_sig.dim()==2: x_sig = x_sig[-1]         # take the most recent signature snapshot
        if x_sig.dim()==1: x_sig = x_sig.unsqueeze(0).repeat(x_ts.size(0),1)  # [N,Dsig]
        h_sig = self.sig(x_sig)
        return self.head(torch.cat([h_ts, h_sig], dim=1))

class LSTMPolicy(nn.Module):
    def __init__(self, in_ts=M_in, in_sig=Dsig, h=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ts, hidden_size=h, num_layers=2, batch_first=True)
        self.sig  = nn.Sequential(nn.Linear(in_sig,64), nn.ReLU(), nn.Linear(64,32))
        self.head = nn.Sequential(nn.Linear(h+32,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_ts, x_sig):
        _, (hn, _) = self.lstm(x_ts)
        h_ts = hn[-1]
        if x_sig.dim()==2: x_sig = x_sig.unsqueeze(0).repeat(x_ts.size(0),1,1)
        h_sig = self.sig(x_sig[:,-1,:])
        return self.head(torch.cat([h_ts, h_sig], dim=1))

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

def make_policy(name):
    if name=="ffn": return FFNPolicy(K).to(device)
    if name=="lstm": return LSTMPolicy().to(device)
    if name=="transformer": return TransformerPolicy().to(device)

# ---------------- kernel-on-signatures (mean field) ----------------
@torch.no_grad()
def _silverman_gamma(W):
    # Bandwidth via Silverman; return gamma = 1/(2 h^2)
    Wc = W - W.mean(dim=0, keepdim=True)
    sigma = (Wc.pow(2).mean().sqrt() + 1e-12).item()
    n = W.shape[0]
    h = 1.06 * sigma * (n ** (-1.0/5.0)) if n > 1 else 1.0
    return 0.5 / (h*h + 1e-12)

@torch.no_grad()
def kernel_mean_at_t(Xvals, Sig_bank, sig_query, gamma=None):
    # Xvals: [N]      Sig_bank: [N, Dsig]   sig_query: [Dsig]
    if gamma is None: gamma = _silverman_gamma(Sig_bank)
    d2 = ((Sig_bank - sig_query.unsqueeze(0))**2).sum(dim=1)   # [N]
    w  = torch.exp(-gamma * d2) + 1e-12
    return (w * Xvals).sum() / w.sum()

# ---------------- simple metrics & losses ----------------
def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    dd = (cum-peak)/np.clip(peak,1e-12,None)
    return float(dd.min())

def compute_loss(X_paths, pi_series, mf_series):
    # You can plug your custom loss by flipping --loss=custom and importing it here
    mode = args.loss
    if mode == "custom":
        import custom_loss  # your separate module
        return custom_loss.compute_loss(X_paths, pi_series, mf_series, dt, 0.1)

    # conservative defaults that read well in a viva:
    if mode == "terminal_log":
        return -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
    if mode == "variance_penalty":
        term = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
        return term + 0.1 * X_paths[:,-1,0].var()
    if mode == "control_quadratic":
        term = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
        return term + 0.1 * (pi_series[:,:,0]**2).mean()
    # default: drawdown-penalised log utility
    term = -(torch.log(torch.clamp(X_paths[:,-1,0], min=1e-8))).mean()
    with torch.no_grad():
        mdd = -max_drawdown(X_paths[:,:,0].mean(dim=0).cpu().numpy())
    return term + 0.1*torch.tensor(mdd, dtype=torch.float32, device=X_paths.device)

# ---------------- train loop ----------------
def train_once():
    print(f"Device={device} | Model={args.model} | Out={args.out}")
    policy = make_policy(args.model)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # initialise mean-field path (all ones), keep a running EMA
    mf = torch.full((N, K+1, 1), X0, device=device)

    losses, mean_XT = [], []

    for it in range(1, args.iters+1):
        # one idio noise per agent; one common noise shared across agents (can extend later)
        dW = torch.randn(N, K, 1, device=device) * math.sqrt(dt)
        dB_common = torch.randn(K, 1, device=device) * math.sqrt(dt)
        dB = dB_common.unsqueeze(0).repeat(N,1,1)

        # --- mean-field update (kernel-on-signatures), gradients detached ---
        with torch.no_grad():
            X_sim = torch.zeros(N, K+1, 1, device=device); X_sim[:,0,0] = X0
            Sig_bank = sig_grid.unsqueeze(0).repeat(N,1,1)  # same signature path for now
            for k in range(K):
                pi_k = policy(series_agents, sig_grid)           # [N,1]
                dX   = pi_k*(args.mu*dt + args.nu*dW[:,k] + args.sigma*dB[:,k])
                X_sim[:,k+1] = X_sim[:,k] + dX

            # conditional mean at each time via Nadaraya–Watson (signature features)
            # note: with *identical* sig per agent this reduces to the batch mean;
            # structure kept so it's trivial to enable multiple common paths later.
            for k in range(K+1):
                mf_val = X_sim[:,k,0].mean()
                mf[:,k,0] = (1 - args.mf_alpha)*mf[:,k,0] + args.mf_alpha*mf_val

        # --- training step (backprop only through the policy) ---
        X_path  = torch.zeros(N, K+1, 1, device=device); X_path[:,0,0] = X0
        actions = torch.zeros(N, K, 1, device=device)
        for k in range(K):
            a_k = policy(series_agents, sig_grid)
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
        if it % 50 == 0 or it == 1:
            print(f"[{args.model}] it={it:>4}  loss={losses[-1]:.6f}  mean_XT={mean_XT[-1]:.4f}")

        if it % args.checkpoint_every == 0 or it == args.iters:
            torch.save({"iter": it, "policy": policy.state_dict()},
                       os.path.join(args.out,"ckpt", f"{args.model}_it{it}.pt"))

    # quick summaries
    XT = X_path[:,:,0].detach().cpu().numpy()
    rets = (XT[:,1:] - XT[:,:-1]) / np.clip(XT[:,:-1], 1e-12, None)
    mean_rets = rets.mean(axis=0)
    sharpe_proxy = float(mean_rets.mean()/(mean_rets.std()+1e-12))
    mdd = -max_drawdown(XT.mean(axis=0))

    pd.DataFrame(losses, columns=["loss"]).to_csv(os.path.join(args.out,"losses.csv"), index=False)
    pd.DataFrame(XT[:,-1], columns=["terminal_wealth"]).to_csv(os.path.join(args.out,"final_wealths.csv"), index=False)
    pd.DataFrame(mean_XT, columns=["mean_XT"]).to_csv(os.path.join(args.out,"mean_XT.csv"), index=False)

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
                   "loss_mode": args.loss, "Sharpe_proxy": sharpe_proxy,
                   "MaxDrawdown_mean_path": mdd}, f, indent=2)

if __name__ == "__main__":
    train_once()
    print("✅ Done:", args.out)