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

- `meta/llama-65b`: ``,
- `meta/llama-2-7b`: `meta-llama/Llama-2-7b-chat-hf`,
- `meta/llama-2-13b`: `meta-llama/Llama-2-13b-chat-hf`
- `meta/llama-2-70b`: `meta-llama/Llama-2-70b-chat-hf`,
- `meta/llama-3-8b`: `meta-llama/Meta-Llama-3-8B-Instruct`,
- `meta/llama-3-70b`: ``,
- `meta/llama-3.1-8b-instruct-turbo`: `meta-llama/Llama-3.1-8B-Instruct`,
- `meta/llama-3.1-70b-instruct-turbo`: `meta-llama/Llama-3.1-70B-Instruct`,
- `meta/llama-3.1-405b-instruct-turbo`,
- `meta/llama-3.2-11b-vision-instruct-turbo`: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `meta/llama-3.2-90b-vision-instruct-turbo`'
- `meta/llama-3.3-70b-instruct-turbo`,

To evaluate predictions

```bash
python evaluate-v1.0.py --predictions <file-path>
```

```python
## squad_meta-llama__Llama-2-13b-chat-hf_20251004-011054.json
{
  "total": 963,
  "exact": 0.20768431983385255,
  "f1": 37.896657753332995,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-2-7b-chat-hf_20251004-035916.json
{
  "total": 963,
  "exact": 0.3115264797507788,
  "f1": 36.81620378706658,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.1-8B-Instruct_20251004-042709.json
{
  "total": 963,
  "exact": 0.0,
  "f1": 44.605185614993744,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Llama-3.2-11B-Vision-Instruct_20251004-033702.json
{
  "total": 963,
  "exact": 0.0,
  "f1": 44.46848936448146,
  "missing_predictions": 0,
  "extra_predictions": 0
}
## squad_meta-llama__Meta-Llama-3-8B-Instruct_20251004-044432.json
{
  "total": 963,
  "exact": 0.0,
  "f1": 43.031860200953616,
  "missing_predictions": 0,
  "extra_predictions": 0
}
```