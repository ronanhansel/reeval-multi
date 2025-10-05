Setting up

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n eval python=3.10 -y
conda activate eval
pip install -r requirements.txt
```

Original models are stored here `~/.cache/huggingface/hub`

On some setup, cache size can be smaller than actually allocated VM sizes, this will change to `\mnt`
which usually has more allocated space. Modify this accordingly to your setup.

```bash
sudo mkdir -p /mnt/hf_cache
sudo chown -R azureuser:azureuser /mnt/hf_cache
echo 'export HF_HOME="/mnt/hf_cache"' >> ~/.bashrc
echo 'export HF_TOKEN=<your-token>' >> ~/.bashrc
```

To remove, clear old models

```bash
rm -rf /mnt/hf_cache/hub/* --verbose
```

Testing models:
`meta/llama`
- `meta/llama-65b`: `upstage/llama-65b-instruct`,
- `meta/llama-2-7b`: `meta-llama/Llama-2-7b-chat-hf`,
- `meta/llama-2-13b`: `meta-llama/Llama-2-13b-chat-hf`
- `meta/llama-2-70b`: `meta-llama/Llama-2-70b-chat-hf`,
- `meta/llama-3-8b`: `meta-llama/Meta-Llama-3-8B-Instruct`,
- `meta/llama-3-70b`: `meta-llama/Meta-Llama-3-70B-Instruct`,
- `meta/llama-3.1-8b-instruct-turbo`: `meta-llama/Llama-3.1-8B-Instruct`,
- `meta/llama-3.1-70b-instruct-turbo`: `meta-llama/Llama-3.1-70B-Instruct`,
- `meta/llama-3.1-405b-instruct-turbo`: `meta-llama/Llama-3.1-405B-Instruct`,
- `meta/llama-3.2-11b-vision-instruct-turbo`: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `meta/llama-3.2-90b-vision-instruct-turbo`: `meta-llama/Llama-3.2-90B-Vision-Instruct`
- `meta/llama-3.3-70b-instruct-turbo`: `meta-llama/Llama-3.3-70B-Instruct`,
`mistralai/mistral`
- `mistralai/mixtral-8x22b`: `mistralai/Mixtral-8x22B-Instruct-v0.1`,
`microsoft/phi`
- `microsoft/phi-3-medium-4k-instruct`: `microsoft/Phi-3-medium-4k-instruct`


```bash
python run_eval.py --model-name <model-name>
```

Batch evaluate multiple models and automatically clear the cache between runs:

```bash
./run_eval_batch.sh upstage/llama-65b-instruct meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-3.2-90B-Vision-Instruct
```

If you omit model names, the script will fall back to a small default list defined at the top of `run_eval_batch.sh`.

To evaluate predictions

```bash
python evaluate-v1.0.py --predictions <file-path>
```

To evaluate all predictions in a folder

```bash
python evaluate-v1.0.py --predictions <folder-path> --all
```

```python
## squad_meta-llama__Llama-2-13b-chat-hf_20251004-011054.json
{
  "total": 963,
  "exact": 61.6822429906542,
  "f1": 78.09362511772163,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-2-70b-chat-hf_20251004-070016.json
{
  "total": 963,
  "exact": 53.271028037383175,
  "f1": 73.16078429168823,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-2-7b-chat-hf_20251004-035916.json
{
  "total": 963,
  "exact": 50.467289719626166,
  "f1": 69.56951849335474,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.1-70B-Instruct_20251004-084829.json
{
  "total": 963,
  "exact": 79.95846313603323,
  "f1": 91.54446065952901,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.1-8B-Instruct_20251004-042709.json
{
  "total": 963,
  "exact": 77.67393561786085,
  "f1": 89.51383702393835,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.2-11B-Vision-Instruct_20251004-033702.json
{
  "total": 963,
  "exact": 79.75077881619937,
  "f1": 90.6362474850219,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.2-90B-Vision-Instruct_20251004-101201.json
{
  "total": 963,
  "exact": 79.95846313603323,
  "f1": 91.69613871183006,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.3-70B-Instruct_20251004-093305.json
{
  "total": 963,
  "exact": 78.92004153686396,
  "f1": 91.2297558602965,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Meta-Llama-3-70B-Instruct_20251004-081114.json
{
  "total": 963,
  "exact": 81.4122533748702,
  "f1": 91.96286103726997,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Meta-Llama-3-8B-Instruct_20251004-044432.json
{
  "total": 963,
  "exact": 79.64693665628245,
  "f1": 90.43791236470378,
  "missing_predictions": 0,
  "extra_predictions": 0
}
```

To perform item-wise evaluation
```bash
python json2csv.py --predictions ./output/
```