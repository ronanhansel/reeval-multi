# reeval-multi

To set up the Python environment:

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n reeval python=3.10 -y
conda activate reeval
pip install -r requirements.txt
```

To install latex-related packages (linux)

```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra cm-super dvipng fonts-liberation
```

If you have problems with jupyter notebook not rendering tqdm correctly on Azure Notebooks, install the following:

```bash
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Download existing external evaluations

```bash
hf download ronanhansel/data-reeval-multi \
    --local-dir ./data-reeval-multi \
    --repo-type dataset
```

- `helm/` contains the code for running the Amortised model on the entire HELM dataset using `embed_meta-llama_Llama-3.1-8B-Instructembed_meta-llama_Llama-3.1-8B-Instruct`, with tuned parameters.
- `hal/` contains the code for running Amortised model on colbench from HAL with `Qwen3-Embedding-8B` along with SAE. `pca_aggregate_survey.ipynb` contains the code for running the model on held out response matrices. Whereas, `sae_beta_irt.ipynb` contains the code for running a single model on `N_samples = 22`
- `item-editor/` contains the **Item-Level Fixing Pipeline** for automatically detecting and fixing Intrinsic Formation Errors (IFEs) in AI agent benchmarks. See [item-editor/README.md](item-editor/README.md) for detailed usage instructions.

Note: To get interpretation, you need to have `OPENAI_KEY_SAE` set in your environment variable.

## Item Editor Pipeline

The item-editor provides a complete pipeline for benchmark defect detection and fixing:

```
Traces → Rubric Evaluation → Verdict Aggregation → Fix Generation → Fix Application
```

### Quick Start

```bash
cd item-editor
pip install -r requirements.txt

# Clone docent dependency
git clone https://github.com/ronanhansel/docent.git
pip install -e docent/docent/ && pip install -e docent/

# Run rubric evaluation
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/*.json \
    --rubric prompts/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

# Aggregate verdicts
python scripts/judge.py \
    --pattern "*.csv" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 -y

# Generate fixes (requires Claude API key)
python scripts/claude_fixer_scicode.py --all-ife

# Apply fixes (requires HAL harness)
python scripts/run_scicode_fixes.py --all --prefix honey_ --docker
```

See [item-editor/README.md](item-editor/README.md) for complete documentation.
