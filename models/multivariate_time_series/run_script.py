import subprocess
from tqdm import tqdm

SEQ_LEN = 4
PRED_LEN = 3

models = ["LSTM", "CHGH", "Autoformer", "DLinear", "Crossformer", "FEDformer", "FiML", "FreTS", "Informer", "Koopa", 
          "LightTS", "Nonstationary_Transformer", "PatchTST", "Reformer", "SegRNN", "TiDE", "Transformer", "TSMixer"]

for model in tqdm(models, desc="Running models", unit="model"):
    cmd = f"python run.py --model {model} --seq_len {SEQ_LEN} --pred_len {PRED_LEN}"
    print(cmd)
    subprocess.run(cmd, shell=True)
