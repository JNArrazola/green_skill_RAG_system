import subprocess
from tqdm import tqdm

SEQ_LEN = 4
PRED_LEN = 3
FUTURE_STEPS = 6

selected_models = ["FEDformer", "Informer", "Reformer"]
selected_models_folders = [f"long_term_forecast_0_{model}_job_demand_region_ftM_sl{SEQ_LEN}_ll1_pl{PRED_LEN}_dm512_nh8_el2_dl2_df2048_expand2_dc4_fc1_eblearned_dtTrue_test_0" for model in selected_models]

if __name__ == "__main__":
    for model_folder, model in tqdm(zip(selected_models_folders, selected_models), desc="Running future predictions", total=len(selected_models), unit="model"):
        cmd = f"python future_prediction.py --seq_len {SEQ_LEN} --pred_len {PRED_LEN} --model_folder {model_folder} --model {model} --future_steps {FUTURE_STEPS}"
        print(cmd)
        subprocess.run(cmd, shell=True)