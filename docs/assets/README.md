# Report figures

PNG files here are used by the main README (dark theme, English labels).

- **Sweep (demo):** `fixtures/demo_sweep_maxprob.csv` — synthetic proxy sweep. Regenerate PNG only:
  ```bash
  python scripts/generate_report_assets.py
  ```
- **Training / eval:** use real `logs.json` and `eval_log.json` from your run (`plot_training_logs.py`, `plot_eval_log.py`, or `scripts/plot_real_bundle.sh`). Do not expect `generate_report_assets.py` to refresh those.
