import subprocess
import os

def run_obs(emb_type, lambda_tau, wd_theta):
    cmd = [
        "~/miniconda3/envs/hal/bin/python", "model/amortized_irt.py",
        "--embedding-type", emb_type,
        "--n-samples", "54",
        "--model-type", "beta",
        "--lambda-tau", str(lambda_tau),
        "--wd-theta", str(wd_theta),
        "--wd-w", "0.0",
        "--epochs", "300"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "model")
    
    print(f"\n\n{'='*50}\nOBSERVING {emb_type.upper()} | lambda={lambda_tau} | wd_theta={wd_theta}\n{'='*50}")
    # Expand the home tilde
    cmd[0] = os.path.expanduser(cmd[0])
    result = subprocess.run(cmd, env=env, capture_output=False)

# First, modify lr_tau in the script to be smaller for stability
import sys
script_path = "model/amortized_irt.py"
with open(script_path, "r") as f:
    text = f.read()
text = text.replace("optimizer_tau = optim.SGD([model.tau_raw], lr=0.5)", "optimizer_tau = optim.SGD([model.tau_raw], lr=0.01)")
with open(script_path, "w") as f:
    f.write(text)

run_obs('pca', 0.0001, 0.1)
run_obs('sae', 0.0001, 0.1)
run_obs('raw', 0.0001, 0.1)
