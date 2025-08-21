
import argparse, json, math, os, re

import numpy as np, pandas as pd



PPYEAR = 252

W_RS = 126



def _coerce_series(x):

    """Accept either wealth or returns; return wealth and returns."""

    s = pd.Series(x).dropna().astype(float)

    if (s<=0).any() or s.median() < 0.2 or s.median() > 5.0:

        # looks like returns; build wealth

        r = s.values

        w = np.cumprod(1.0 + r)

    else:

        w = s.values

        r = np.r_[np.nan, w[1:]/w[:-1]-1.0]

    return pd.Series(w).dropna().reset_index(drop=True), pd.Series(r).dropna().reset_index(drop=True)



def _dd(wealth):

    peak = np.maximum.accumulate(wealth)

    dd = wealth/peak - 1.0

    return dd.min(), dd



def _metrics_from_wealth(wealth):

    w, r = _coerce_series(wealth)

    mu = r.mean()*PPYEAR

    vol = r.std(ddof=1)*np.sqrt(PPYEAR)

    sharpe = 0.0 if vol==0 else mu/vol

    mdd, _ = _dd(w.values)

    years = len(r)/PPYEAR

    cagr = (w.iloc[-1])**(1/years) - 1 if years>0 else np.nan

    calmar = (cagr/abs(mdd)) if (mdd<0) else np.nan

    var95 = np.quantile(r, 0.05)

    cvar95 = r[r<=var95].mean()

    return dict(sharpe=sharpe, ann_return=mu, ann_vol=vol,

                max_dd=float(mdd), calmar=float(calmar) if isinstance(calmar,float) or not np.isnan(calmar) else np.nan,

                var_95=float(var95), cvar_95=float(cvar95), T=len(r), wealth_end=float(w.iloc[-1]))



def _read_any_csv(path):

    df = pd.read_csv(path)

    # pick first numeric column if needed

    num = df.select_dtypes(include=[np.number])

    if num.shape[1]==0:

        # try last column after coercion

        num = df.apply(pd.to_numeric, errors='coerce')

        num = num.select_dtypes(include=[np.number])

    s = num.iloc[:,0]

    return s



def scan_runs(root):

    out = []

    if not os.path.isdir(root): return out

    for d in sorted(os.listdir(root)):

        run_dir = os.path.join(root, d)

        if not os.path.isdir(run_dir): continue

        # label arch/tune from folder name

        name = d.lower()

        arch = 'ffn' if 'ffn' in name else ('lstm' if 'lstm' in name else ('transformer' if 'transform' in name else 'na'))

        tune = 'bo' if re.search(r'\bbo\b|_bo_|-bo', name) else 'base'

        # find a wealth file

        cand = []

        for fn in os.listdir(run_dir):

            if re.search(r'xt.*\.csv|wealth.*\.csv', fn, flags=re.I):

                cand.append(os.path.join(run_dir, fn))

        if not cand: 

            # sometimes saved under nested dirs

            for sub in ['.', 'eval', 'outputs']:

                p = os.path.join(run_dir, sub)

                if os.path.isdir(p):

                    for fn in os.listdir(p):

                        if re.search(r'xt.*\.csv|wealth.*\.csv', fn, flags=re.I):

                            cand.append(os.path.join(p, fn))

        if not cand: 

            continue

        # read first viable

        s = _read_any_csv(cand[0])

        m = _metrics_from_wealth(s.values)

        m.update(run=d, arch=arch, tune=tune, path=cand[0])

        out.append(m)

    return pd.DataFrame(out)



def load_benchmarks(eval_root):

    """Use the benchmark series you produced; fall back to cash if missing."""

    benches = {}

    for tag, fname in [

        ('Cash', None),

        ('Equal-Weight', 'equal_weight_series.csv'),

        ('MV-Rolling-252', 'mv_rolling_W252_series.csv'),

        ('MV-Static', 'mean_variance_series.csv'),

        ('VaRmin-95', 'varmin_a0.05_series.csv')

    ]:

        if fname is None:

            wealth = pd.Series(np.ones(2))  # flat line for metrics; handled as cash

            benches[tag] = wealth

            continue

        fpath = os.path.join(eval_root, fname)

        if os.path.exists(fpath):

            benches[tag] = _read_any_csv(fpath)

    return benches



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-root", required=True, help="Where to write figures/tables/csv")

    ap.add_argument("--search-roots", required=True, help="Comma-separated list of run roots to scan")

    ap.add_argument("--version", default="fast1000")

    args = ap.parse_args()



    os.makedirs(args.eval_root, exist_ok=True)

    tables_dir = os.path.join(args.eval_root, "tables"); os.makedirs(tables_dir, exist_ok=True)



    # scan runs

    run_roots = [s.strip() for s in args.search_roots.split(",") if s.strip()]

    frames = [scan_runs(r) for r in run_roots]

    runs = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)

    if not len(runs):

        print("WARN: no wealth files found under:", run_roots)

    else:

        runs.to_csv(os.path.join(args.eval_root, f"summary_metrics_all_{args.version}.csv"), index=False)



    # aggregate for the tables used in the paper

    if len(runs):

        # mean by arch×tune

        agg = runs.groupby(["arch","tune"], as_index=False)[["sharpe","ann_return","ann_vol","calmar","max_dd","var_95","cvar_95"]].mean()

        agg.to_csv(os.path.join(args.eval_root, f"arch_tune_summary_{args.version}.csv"), index=False)

        # LaTeX

        tex = agg.to_latex(index=False, float_format="%.3f", caption=f"Mean metrics by architecture and tuning ({args.version})",

                           label=f"tab:arch_tune_summary_{args.version}")

        with open(os.path.join(args.eval_root, f"arch_tune_summary_{args.version}.tex"), "w") as f: f.write(tex)



        # best per arch×tune

        idx = runs.groupby(["arch","tune"])["sharpe"].idxmax()

        best = runs.loc[idx].sort_values(["arch","tune"])

        best.to_csv(os.path.join(args.eval_root, f"arch_tune_best_{args.version}.csv"), index=False)

        tex = best[["arch","tune","run","sharpe","ann_return","ann_vol","max_dd","calmar"]].to_latex(

            index=False, float_format="%.3f", caption=f"Best runs per architecture and tuning ({args.version})",

            label=f"tab:arch_tune_best_{args.version}"

        )

        with open(os.path.join(args.eval_root, f"arch_tune_best_{args.version}.tex"), "w") as f: f.write(tex)



    # benchmarks metrics

    benches = load_benchmarks(args.eval_root)

    bench_rows = []

    for name, s in benches.items():

        m = _metrics_from_wealth(s.values)

        m.update(bench=name)

        bench_rows.append(m)

    if bench_rows:

        bdf = pd.DataFrame(bench_rows)

        bdf.to_csv(os.path.join(args.eval_root, f"benchmark_metrics_{args.version}.csv"), index=False)

        tex = bdf[["bench","sharpe","ann_return","ann_vol","max_dd","var_95","cvar_95"]].to_latex(

            index=False, float_format="%.3f", caption=f"Benchmark metrics ({args.version})",

            label=f"tab:bench_metrics_{args.version}"

        )

        with open(os.path.join(args.eval_root, f"benchmark_metrics_{args.version}.tex"), "w") as f: f.write(tex)



if __name__ == "__main__":

    main()

