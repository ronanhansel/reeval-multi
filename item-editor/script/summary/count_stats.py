import os, glob, csv

print("--- VERDICTS ---")
for b in ["colbench_backend_programming", "scicode", "scienceagentbench", "corebench_hard", "usaco"]:
    v_file = f"/Users/ronan/Developer/agent-eval/item-editor/eval_response_matrix/pre-revision/{b}/verdicts/verdict_original.csv"
    if os.path.exists(v_file):
        with open(v_file) as f:
            r = list(csv.reader(f))
            if len(r) > 1:
                v = r[1][1:]
                print(f"{b} -> defects (1): {v.count('1')}, genuine (0): {v.count('0')}")

print("\n--- FIXES ---")
fixes_dir = "/Users/ronan/Developer/agent-eval/item-editor/result/fixes"
for b in os.listdir(fixes_dir):
    p = os.path.join(fixes_dir, b)
    if os.path.isdir(p):
        env = len(glob.glob(f"{p}/**/*.json", recursive=True))
        env_only = len(glob.glob(f"{p}/**/env_override.json", recursive=True))
        inst = len(glob.glob(f"{p}/**/instruction_override.json", recursive=True))
        eval = len(glob.glob(f"{p}/**/evaluation_override.json", recursive=True))
        deps = len(glob.glob(f"{p}/**/dependency_override.json", recursive=True))
        inputs = len(glob.glob(f"{p}/**/input_override.json", recursive=True))
        sim = len(glob.glob(f"{p}/**/simulated_user_override.json", recursive=True))
        total = env_only + inst + eval + deps + inputs + sim
        print(f"{b} -> Total JSONOverrides: {total}, Env: {env_only}, Inst: {inst}, Eval: {eval}, Deps: {deps}, Inputs: {inputs}, Sim: {sim}")
