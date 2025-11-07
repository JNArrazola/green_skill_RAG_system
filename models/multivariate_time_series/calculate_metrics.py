from metrics import compute_metrics
import os
import numpy as np
import re

rgx = "(?<=forecast_0)(.*?)(?=job_demand)"
FOLDER = "results/seq_4_len_3"
MD = "../../doc/metric_results_4_3.md"

if not os.path.exists(MD):
    with open(MD, "w") as f:
        f.write("# Metric Results for Sequence Length 4, Prediction Length 3\n")
        f.write("| Model | MAE | RMSE | sMAPE | rRMSE |\n")
        f.write("|-------|-----|------|-------|-------|\n")

for folder in os.listdir(FOLDER):
    pred = np.load(os.path.join(FOLDER, folder, "pred.npy"))
    true = np.load(os.path.join(FOLDER, folder, "true.npy"))
    metrics = compute_metrics(true, pred)
    model_name = re.search(rgx, folder).group(0)[1:-1]
    with open(MD, "a") as f:
        f.write(f"| {model_name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['smape']:.4f} | {metrics['rrmse']:.4f} |\n")