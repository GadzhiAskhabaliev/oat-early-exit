# Report figures

PNG files here are generated for the main README (dark theme, English labels).

- **Regenerate (optional):** from repo root run  
  `python scripts/generate_report_assets.py`  
  This writes synthetic fixtures under `fixtures/` and rebuilds the three `figure_*.png` files. Swap in your real `logs.json`, `eval_log.json`, or sweep CSV paths and call `plot_training_logs.py`, `plot_eval_log.py`, or `plot_sweep_csv.py` directly for publication-quality plots.
