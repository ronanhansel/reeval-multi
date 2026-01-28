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
