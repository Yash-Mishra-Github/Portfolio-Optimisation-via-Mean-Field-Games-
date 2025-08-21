# Portfolio Optimisation via Mean Field Games

This repository accompanies my MSc dissertation:

**Signatured Deep Fictitious Play (Sig-DFP) for Portfolio Optimisation with Common Noise**

---

##  Repository Structure

src/ # Python source code
│ ├── train_quick.py # Stage I: baseline trainer
│ ├── train_mfg.py # Stage II: trainer with signatures + EMA mean field
│ ├── sig_features.py # Signature feature extractor
│ ├── custom_loss.py # Custom loss functions
│ └── lemma_fix/
│ └── mean_field_fix.py
slurm/ # SLURM job submission scripts
│ ├── run_quick_base.slurm
│ ├── run_quick_tuned.slurm
│ ├── run_stage2_base.slurm
│ ├── run_stage2_tuned.slurm
│ └── run_eval.slurm
data/
│ └── processed_market_data.csv # Small demo dataset
README.md
.gitignore


---

