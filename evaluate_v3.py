
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Fabian-ready one-shot evaluator for Sig-DFP + benchmarks.

This is a path-tuned variant that defaults to /users/mishray/V3.
You can override all paths via CLI flags or environment variables.

What it does (one shot):
- Scans one or more outputs roots for Sig-DFP runs that contain XT_series.csv
- For each run:
    • Wealth mean±σ, Drawdown curve on mean wealth
    • Sharpe(t) across paths, Terminal wealth histogram
    • Risk–Return scatter (per-path μ vs σ)
    • Optional training loss plot if losses.csv exists
    • Per-run CSVs for sharpe_t, mdd_t
- Benchmarks on your processed market data (rolling window):
    • Equal-Weight (EW)
    • Mean-Variance (MV) with covariance shrinkage + long-only projection
    • VaR-minimization via candidate sweep (EW, RP, MinVar, MV)
    • Saves returns, weights, rolling sharpe, drawdown, plots
- Cross-run leaderboard and Top-5 mean-wealth overlay
- Unique results folder per invocation to avoid overwrites
- Robust logging and guards for SVD/NaNs

Usage example:
    python evaluate_v3.py \
      --outputs-root /users/mishray/V3/outputs /users/mishray/outputs \
      --data-file /users/mishray/V3/data/processed_market_data.csv \
      --window 126 \
      --res-root /users/mishray/V3/results \
      --tag nightly

Notes:
- XT_series.csv shape expected: [N_paths, T+1] of wealth trajectories (≥0).
- If any file/dir is missing, the script logs and continues to the next item.
"""
import os, sys, glob, json, math, argparse, traceback, logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- Utilities & Logging ----------------------------

def makedirs(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_dir: str):
    makedirs(log_dir)
    log_file = os.path.join(log_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging to %s", log_file)

def plot(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=220)
    plt.close(fig)

def safe_div(a, b, eps=1e-12):
    return a / (np.sign(b) * np.maximum(np.abs(b), eps))

def nan_guard(x, fill=0.0):
    if isinstance(x, np.ndarray):
        x = np.where(np.isfinite(x), x, fill)
    else:
        x = fill if (x is None or not math.isfinite(x)) else x
    return x


# ---------------------------- Metrics & Curves ------------------------------

def returns_from_xt(XT: np.ndarray) -> np.ndarray:
    """XT shape [N_paths, T+1]; returns R shape [N_paths, T]."""
    prev = np.clip(XT[:, :-1], 1e-12, None)
    R = (XT[:, 1:] - XT[:, :-1]) / prev
    return R

def sharpe_over_time(R: np.ndarray) -> np.ndarray:
    """Across paths, per t: mean/ std (no annualization)."""
    mu = np.nanmean(R, axis=0)
    sd = np.nanstd(R, axis=0) + 1e-12
    return mu / sd

def mean_drawdown_curve(wealth: np.ndarray) -> np.ndarray:
    """Drawdown curve on a single wealth curve (e.g., mean over paths)."""
    peak = np.maximum.accumulate(wealth)
    return wealth / np.clip(peak, 1e-12, None) - 1.0

def var_percentile(flat_returns: np.ndarray, q=0.05) -> float:
    return float(np.nanpercentile(flat_returns, 100*q))

def expected_shortfall(flat_returns: np.ndarray, q=0.05) -> float:
    """ES (a.k.a. CVaR) at level q using historical average of tail."""
    if flat_returns.size == 0:
        return float('nan')
    v = np.nanpercentile(flat_returns, 100*q)
    tail = flat_returns[flat_returns <= v]
    if tail.size == 0:
        return float(v)
    return float(np.nanmean(tail))

def max_drawdown(wealth: np.ndarray) -> float:
    dd = mean_drawdown_curve(wealth)
    return float(dd.min())

def terminal_stats(XT: np.ndarray) -> Dict[str, float]:
    term = XT[:, -1]
    return {
        "terminal_mean": float(np.nanmean(term)),
        "terminal_std":  float(np.nanstd(term)),
        "terminal_var":  float(np.nanvar(term)),
        "terminal_log_mean": float(np.nanmean(np.log(np.clip(term, 1e-12, None)))),
    }


# ---------------------------- Plot helpers ----------------------------------

def plot_wealth_band(name: str, XT: np.ndarray, outdir: str):
    curve = np.nanmean(XT, axis=0)
    band = np.nanstd(XT, axis=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(curve, label="mean wealth")
    ax.fill_between(range(len(curve)), curve - band, curve + band, alpha=0.25)
    ax.set_title(f"Wealth mean±σ — {name}")
    ax.grid(True, alpha=0.3); ax.legend()
    plot(fig, os.path.join(outdir, "wealth_curve.png"))

def plot_sharpe_curve(name: str, SRt: np.ndarray, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(SRt)
    ax.set_title(f"Sharpe(t) — {name}")
    ax.grid(True, alpha=0.3)
    plot(fig, os.path.join(outdir, "sharpe_curve.png"))

def plot_dd_curve(name: str, DD: np.ndarray, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(DD)
    ax.set_title(f"Drawdown — {name}")
    ax.grid(True, alpha=0.3)
    plot(fig, os.path.join(outdir, "mdd_curve.png"))

def plot_terminal_hist(XT: np.ndarray, outdir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(XT[:, -1], bins=40)
    ax.set_title("Terminal wealth")
    ax.grid(True, alpha=0.3)
    plot(fig, os.path.join(outdir, "terminal_wealth_hist.png"))

def plot_risk_return_scatter(R: np.ndarray, outdir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(R.std(1), R.mean(1), s=8, alpha=0.6)
    ax.set_xlabel("σ"); ax.set_ylabel("μ")
    ax.set_title("Risk–Return (paths)")
    ax.grid(True, alpha=0.3)
    plot(fig, os.path.join(outdir, "risk_return_scatter.png"))

def plot_convergence_if_any(run_dir: str, outdir: str):
    loss_csv = os.path.join(run_dir, "losses.csv")
    if not os.path.isfile(loss_csv):
        return False
    try:
        df = pd.read_csv(loss_csv)
        for col in ("loss","train_loss","val_loss"):
            if col in df.columns:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(df[col].values)
                ax.set_title(f"Training {col.replace('_',' ')} (as logged)")
                ax.grid(True, alpha=0.3)
                plot(fig, os.path.join(outdir, f"{col}.png"))
        return True
    except Exception as e:
        logging.warning("Failed to plot convergence for %s: %s", run_dir, e)
    return False


# ---------------------------- Sig-DFP Evaluator -----------------------------

def evaluate_sigdfp_runs(outputs_roots: List[str], res_root: str) -> pd.DataFrame:
    rows = []
    all_runs = []
    for root in outputs_roots:
        if not os.path.isdir(root):
            logging.info("Outputs root not found (skip): %s", root); continue
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d): 
                continue
            xt = os.path.join(d, "XT_series.csv")
            if os.path.isfile(xt):
                all_runs.append((root, d))

    if not all_runs:
        logging.warning("No Sig-DFP runs with XT_series.csv found under: %s", outputs_roots)
        return pd.DataFrame(rows)

    for root, run_dir in all_runs:
        name = os.path.basename(run_dir.rstrip("/"))
        outdir = makedirs(os.path.join(res_root, "sigdfp", name))
        try:
            XT = pd.read_csv(os.path.join(run_dir, "XT_series.csv")).values
            if XT.ndim != 2 or XT.shape[1] < 2:
                raise ValueError(f"XT_series.csv has unexpected shape {XT.shape}")
            R = returns_from_xt(XT)
            SRt = sharpe_over_time(R)
            curve = np.nanmean(XT, axis=0)
            DD = mean_drawdown_curve(curve)
            flat = R.reshape(-1)

            # Save metrics
            pd.DataFrame({"sharpe_t": nan_guard(SRt)}).to_csv(os.path.join(outdir, "sharpe_over_time.csv"), index=False)
            pd.DataFrame({"mdd_t": nan_guard(DD)}).to_csv(os.path.join(outdir, "mdd_over_time.csv"), index=False)
            # Plots
            plot_wealth_band(name, XT, outdir)
            plot_sharpe_curve(name, SRt, outdir)
            plot_dd_curve(name, DD, outdir)
            plot_terminal_hist(XT, outdir)
            plot_risk_return_scatter(R, outdir)
            plot_convergence_if_any(run_dir, outdir)

            # Summary row
            term = terminal_stats(XT)
            row = {
                "model": "Sig-DFP",
                "run": name,
                "sharpe_overall": float(np.nanmean(flat) / (np.nanstd(flat) + 1e-12)),
                "mdd": float(DD.min()),
                "var95": var_percentile(flat, 0.05),
                "es95": expected_shortfall(flat, 0.05),
                "mean_log_utility": term["terminal_log_mean"],
                "terminal_variance": term["terminal_var"],
                "ctrl_proxy": float(np.nanmean((XT[:, 1:] - XT[:, :-1])**2)),
            }
            rows.append(row)
            logging.info("Evaluated Sig-DFP run: %s", name)
        except Exception as e:
            logging.error("Failed on run %s: %s\n%s", name, e, traceback.format_exc())

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(res_root, "sigdfp_summaries.csv"), index=False)
    return df


# ---------------------------- Benchmarks ------------------------------------

def _as_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic to build a wide returns table from various CSV layouts."""
    dfx = df.copy()
    # WRDS style ['date','TICKER','RET'] -> pivot
    cols = {c.lower(): c for c in dfx.columns}
    lower = [c.lower() for c in dfx.columns]
    dfx.columns = lower
    if {"date","ticker","ret"}.issubset(set(dfx.columns)):
        dfx["date"] = pd.to_datetime(dfx["date"])
        wide = dfx.pivot_table(index="date", columns="ticker", values="ret").sort_index()
        return wide.dropna(how="all")

    # Else: assume wide prices or wide returns
    if "date" in dfx.columns:
        dfx["date"] = pd.to_datetime(dfx["date"])
        dfx = dfx.set_index("date").sort_index()

    num = dfx.select_dtypes(include=[np.number])
    # If already returns: median magnitude < 0.5 and signs mixed
    if (num.abs().median().median() < 0.5) and ((num < 0).sum().sum() > 0):
        returns = num.dropna(how="all")
    else:
        returns = num.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, how="all").fillna(0.0)
    returns = returns.loc[:, returns.std() > 0]
    return returns

def load_returns_table(data_file: Optional[str]) -> Optional[pd.DataFrame]:
    if data_file is None:
        logging.warning("No data file provided for benchmarks; skipping benchmarks.")
        return None
    if not os.path.isfile(data_file):
        logging.error("Provided data file not found: %s", data_file)
        return None
    try:
        df = pd.read_csv(data_file)
        R = _as_returns(df)
        if R is None or R.empty:
            logging.error("Could not derive returns table from %s", data_file)
            return None
        R = R.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        R = R.clip(lower=-0.5, upper=0.5)  # tame outliers to reduce SVD issues
        logging.info("Loaded returns table shape=%s from %s", R.shape, data_file)
        return R
    except Exception as e:
        logging.error("Failed to load returns from %s: %s", data_file, e)
        return None

def project_long_only(w: np.ndarray, eps=1e-9) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w if s <= eps else (w / s)

def shrink_cov(S: np.ndarray, lam: float = 0.10) -> np.ndarray:
    S = 0.5 * (S + S.T)  # symmetrize
    diag = np.diag(np.diag(S))
    S_sh = (1.0 - lam) * S + lam * diag
    jitter = 1e-8 * np.trace(S_sh) / max(1, S_sh.shape[0])
    S_sh = S_sh + jitter * np.eye(S_sh.shape[0])
    return S_sh

def mean_variance_weights(mu: np.ndarray, S: np.ndarray, long_only=True) -> np.ndarray:
    try:
        Sinv = np.linalg.pinv(S, rcond=1e-6)
    except Exception:
        Sinv = np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]), rcond=1e-6)
    w = Sinv @ mu
    if long_only:
        w = project_long_only(w)
    else:
        s = w.sum(); w = w / (s if abs(s) > 1e-12 else 1.0)
    return w

def risk_parity_weights(S: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    w = 1.0 / vol
    return project_long_only(w)

def min_variance_weights(S: np.ndarray) -> np.ndarray:
    ones = np.ones(S.shape[0])
    try:
        Sinv = np.linalg.pinv(S, rcond=1e-6)
    except Exception:
        Sinv = np.linalg.pinv(S + 1e-6 * np.eye(S.shape[0]), rcond=1e-6)
    w = Sinv @ ones
    return project_long_only(w)

def ew_weights(n: int) -> np.ndarray:
    return np.ones(n) / float(n)

def portfolio_returns(R: pd.DataFrame, W: np.ndarray) -> np.ndarray:
    if W.ndim == 1:
        return (R.values @ W).astype(float)
    else:
        T = min(len(R), len(W))
        return (R.values[:T, :] * W[:T, :]).sum(axis=1).astype(float)

def wealth_from_returns(r: np.ndarray, x0: float = 1.0) -> np.ndarray:
    w = np.empty(len(r) + 1, dtype=float)
    w[0] = x0
    for i in range(len(r)):
        w[i+1] = max(1e-12, w[i] * (1.0 + float(r[i])))
    return w

def rolling_backtest(R: pd.DataFrame, window: int = 126,
                     strategy: str = "EW",
                     shrink: float = 0.10,
                     long_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    Rv = R.values
    T, N = Rv.shape
    rets = []
    Wt_list = []
    W_prev = np.ones(N)/N  # warm start with EW

    for t in range(T):
        if t < window:
            Wt = W_prev
            rets.append(float((Rv[t:t+1, :] @ Wt).sum()))
            Wt_list.append(Wt)
            continue

        hist = Rv[t-window:t, :]
        mu = np.nanmean(hist, axis=0)
        S = np.cov(hist, rowvar=False)
        S = shrink_cov(S, lam=shrink)

        if strategy == "EW":
            Wt = ew_weights(N)
        elif strategy == "MV":
            Wt = mean_variance_weights(mu, S, long_only=long_only)
        elif strategy == "VARMIN":
            cand = [
                ("EW", ew_weights(N)),
                ("RP", risk_parity_weights(S)),
                ("MINVAR", min_variance_weights(S)),
                ("MV", mean_variance_weights(mu, S, long_only=True)),
            ]
            best_w, best_v = None, +1e9
            for name, w in cand:
                port_hist = hist @ w
                v = np.nanpercentile(port_hist, 5.0)
                if v < best_v:
                    best_v = v; best_w = w
            Wt = best_w
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        W_prev = Wt
        rets.append(float((Rv[t:t+1, :] @ Wt).sum()))
        Wt_list.append(Wt)

    return np.array(rets), np.stack(Wt_list, axis=0)

def rolling_metrics(r: np.ndarray, win: int = 63) -> Dict[str, np.ndarray]:
    sr = np.full(len(r), np.nan)
    w = wealth_from_returns(r)
    peak = np.maximum.accumulate(w)
    dd = w / np.clip(peak, 1e-12, None) - 1.0
    for t in range(win, len(r)):
        window = r[t-win:t]
        mu = np.nanmean(window); sd = np.nanstd(window) + 1e-12
        sr[t] = mu / sd
    return {"rolling_sharpe": sr, "drawdown": dd}

def run_benchmarks(data_file: Optional[str], res_root: str, window: int = 126) -> pd.DataFrame:
    R = load_returns_table(data_file)
    if R is None or R.empty:
        return pd.DataFrame([])

    bench_specs = [
        ("EW",      {"strategy": "EW"}),
        ("MV",      {"strategy": "MV", "shrink": 0.10, "long_only": True}),
        ("VARMIN",  {"strategy": "VARMIN", "shrink": 0.10}),
    ]

    rows = []
    outdir_root = makedirs(os.path.join(res_root, "benchmarks"))
    for name, kw in bench_specs:
        try:
            ret_series, Wt = rolling_backtest(R, window=window, **kw)
            wealth = wealth_from_returns(ret_series, x0=1.0)
            flat = ret_series
            # Per-benchmark outputs
            outdir = makedirs(os.path.join(outdir_root, name))
            pd.DataFrame({"ret": flat}).to_csv(os.path.join(outdir, "returns.csv"), index=False)
            pd.DataFrame(Wt, columns=R.columns).to_csv(os.path.join(outdir, "weights.csv"), index=False)
            # Metrics & plots
            mets = rolling_metrics(flat, win=63)
            pd.DataFrame({"rolling_sharpe": mets["rolling_sharpe"]}).to_csv(os.path.join(outdir, "rolling_sharpe.csv"), index=False)
            pd.DataFrame({"drawdown": mets["drawdown"]}).to_csv(os.path.join(outdir, "drawdown_curve.csv"), index=False)
            # Plots
            fig, ax = plt.subplots(figsize=(8,3.5)); ax.plot(wealth); ax.set_title(f"Benchmark wealth — {name}"); ax.grid(True, alpha=0.3)
            plot(fig, os.path.join(outdir, "wealth_curve.png"))
            fig, ax = plt.subplots(figsize=(7,3)); ax.plot(mets["rolling_sharpe"]); ax.set_title(f"Rolling Sharpe — {name}"); ax.grid(True, alpha=0.3)
            plot(fig, os.path.join(outdir, "rolling_sharpe.png"))
            fig, ax = plt.subplots(figsize=(7,3)); ax.plot(mets["drawdown"]); ax.set_title(f"Drawdown — {name}"); ax.grid(True, alpha=0.3)
            plot(fig, os.path.join(outdir, "drawdown_curve.png"))

            row = {
                "model": f"BM-{name}",
                "run": name,
                "sharpe_overall": float(np.nanmean(flat) / (np.nanstd(flat) + 1e-12)),
                "mdd": float(np.nanmin(mets["drawdown"])),
                "var95": float(np.nanpercentile(flat, 5.0)),
                "es95": float(expected_shortfall(flat, 0.05)),
                "terminal_variance": float(np.nanvar(wealth[-1:])),  # trivial
                "mean_log_utility": float(np.nanmean(np.log(np.clip(wealth[-1], 1e-12, None)))),
            }
            rows.append(row)
            logging.info("Finished benchmark %s", name)
        except Exception as e:
            logging.error("Benchmark %s failed: %s\n%s", name, e, traceback.format_exc())

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(res_root, "benchmark_summaries.csv"), index=False)
    return df


# ---------------------------- Cross-run visuals ------------------------------

def plot_top5_mean_wealth(sig_df: pd.DataFrame, outputs_roots: List[str], res_root: str):
    if sig_df is None or sig_df.empty:
        return
    df = sig_df.sort_values("sharpe_overall", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    k = 0
    for nm in df["run"].tolist():
        run_dir = None
        for root in outputs_roots:
            cand = os.path.join(root, nm, "XT_series.csv")
            if os.path.isfile(cand):
                run_dir = os.path.dirname(cand); break
        if run_dir is None:
            continue
        XT = pd.read_csv(os.path.join(run_dir, "XT_series.csv")).values
        ax.plot(np.nanmean(XT, axis=0), label=nm)
        k += 1
        if k >= 5: break
    ax.legend(); ax.set_title("Top-5 mean wealth (Sig-DFP)"); ax.grid(True, alpha=0.3)
    plot(fig, os.path.join(res_root, "top5_mean_wealth.png"))

def combined_leaderboard(sig_df: pd.DataFrame, bm_df: pd.DataFrame, res_root: str):
    if sig_df is None: sig_df = pd.DataFrame([])
    if bm_df is None: bm_df = pd.DataFrame([])
    df = pd.concat([sig_df, bm_df], ignore_index=True, sort=False)
    if df.empty:
        logging.warning("No results to summarize."); return
    df["rank"] = df["sharpe_overall"].rank(ascending=False, method="dense")
    df = df.sort_values(["rank", "mdd"], ascending=[True, True])
    df.to_csv(os.path.join(res_root, "all_summaries.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = df["model"] + ":" + df["run"]
    ax.bar(range(len(df)), df["sharpe_overall"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe (overall)")
    ax.set_title("Leaderboard: Sig-DFP vs Benchmarks")
    ax.grid(True, axis="y", alpha=0.3)
    plot(fig, os.path.join(res_root, "leaderboard_sharpe.png"))


# ---------------------------- CLI & Main ------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", nargs="+",
                    default=[os.path.expanduser("/users/mishray/V3/outputs"),
                             os.path.expanduser("/users/mishray/outputs"),
                             os.path.expanduser("/users/mishray/outputs_V2")],
                    help="One or more directories to scan for Sig-DFP runs.")
    ap.add_argument("--data-file", type=str,
                    default=os.path.expanduser("/users/mishray/V3/data/processed_market_data.csv"),
                    help="CSV for benchmarks (processed market data).")
    ap.add_argument("--window", type=int, default=126,
                    help="Rolling window length for benchmarks.")
    ap.add_argument("--res-root", type=str,
                    default=os.path.expanduser("/users/mishray/V3/results"),
                    help="Root directory to write evaluation outputs.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional text tag appended to result folder name.")
    return ap.parse_args()

def main():
    args = parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    res_root = os.path.join(os.path.expanduser(args.res_root),
                            f"results_eval_{nowstamp()}{tag}")
    makedirs(res_root)
    setup_logging(res_root)
    logging.info("Results root: %s", res_root)
    logging.info("Outputs roots: %s", args.outputs_root)
    if args.data_file:
        logging.info("Benchmarks data file: %s", args.data_file)

    sig_df = evaluate_sigdfp_runs(args.outputs_root, res_root)
    bm_df = run_benchmarks(args.data_file, res_root, window=args.window)
    plot_top5_mean_wealth(sig_df, args.outputs_root, res_root)
    combined_leaderboard(sig_df, bm_df, res_root)

    meta = {
        "outputs_roots": args.outputs_root,
        "data_file": args.data_file,
        "window": args.window,
        "results_root": res_root,
        "timestamp": nowstamp(),
        "sigdfp_runs_evaluated": 0 if sig_df is None else int(len(sig_df)),
        "benchmarks_evaluated": 0 if bm_df is None else int(len(bm_df)),
    }
    with open(os.path.join(res_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logging.info("DONE. See %s", res_root)
    print("Saved results to:", res_root)

if __name__ == "__main__":
    main()
