# Fix for capsule-9137200 (corebench_hard)

## Problem Diagnosis

This task requires agents to test the computational reproducibility of a Chinese Named Entity Recognition (NER) model (PGAT - Polymorphic Graph Attention Network). The task asks agents to:
1. Install dependencies (torch, transformers, etc.)
2. Run `PGAT/main.py` in test mode
3. Report precision (p), recall (r), and F1 score from the test output

**Root Cause**: The original Code Ocean capsule was designed to run inside Docker with volume mounts that map the capsule's `data` and `results` directories to absolute paths `/data` and `/results` in the container. The code in `NERController.py` has hardcoded absolute paths:

```python
self.result_path = r'/results/'
self.emb_dic_path = r'/data/cache/emb_dic/'
self.input_data_path = r'/data/cache/input_data/'
self.variable_path = r'/data/cache/variable/'
self.word2vec_path = r'/data/cache/word2vec/'
self.inf_path = r'/data/cache/inf/'
```

In the HAL benchmark environment:
- The capsule is staged at `/workspace/environment/`
- The agent runs code from `/workspace/environment/code/PGAT/`
- There are no Docker volume mounts to `/data` or `/results`
- The agent cannot create symlinks at the root level without special permissions

**Evidence from Model Logs**:
- Multiple agents encountered `FileNotFoundError: [Errno 2] No such file or directory: './data/cache/variable/bert/resume'`
- Agents also hit `PermissionError` when the code tried to create directories under `/data/...` or `/results/`
- All 4 models (gpt-4.1-04-14, o3-04-16, o4-mini-04-16 x2) failed with the same infrastructure barrier
- The pretrained model checkpoints ARE present in the capsule at `data/cache/variable/bert/` - they just can't be accessed due to path mismatches

## Fix Applied

**Type**: Input Clarification (input_override.json)

**Solution**: Provide a clarification in the task prompt that explains the path mismatch issue. The agent must understand and modify the hardcoded paths before running the code:

```json
{
  "clarifications": [
    "Note: The code in classes/NERController.py contains hardcoded absolute paths (e.g., '/data/cache/...', '/results/') that assume Docker volume mounts which are not present in this environment. You will need to modify these paths to relative paths (e.g., '../../data/cache/...', '../../results/') before running the code from the code/PGAT directory."
  ]
}
```

## Why This Fix is Appropriate

1. **This is purely an infrastructure clarification**: The original capsule's Docker setup (`REPRODUCING.md`) shows it was designed with volume mounts:
   ```shell
   docker run ... \
     --volume "$PWD/data":/data \
     --volume "$PWD/code":/code \
     --volume "$PWD/results":/results \
     ...
   ```
   The HAL environment doesn't use these volume mounts, so agents need to know paths require adjustment.

2. **The fix does NOT nerf the question**: Agents still need to:
   - Read and understand the README to learn how to run the code
   - Install the required dependencies (torch, transformers, matplotlib, sklearn)
   - **Modify the hardcoded paths in NERController.py** (the clarification tells them what to fix, not how)
   - Determine the correct command-line arguments for test mode
   - Choose which dataset to test on (the expected results are from the `weibo_all` dataset)
   - Parse the output to extract precision, recall, and F1 values

3. **All required data is present**: The pretrained model checkpoints (431-462 MB each for resume, weibo_all, ontonote, ecommerce) are included in the capsule at `data/cache/variable/bert/`. The clarification simply helps agents understand why the paths don't work as-is.

4. **This is a reasonable accommodation**: Without this clarification, agents would need to:
   - Recognize the hardcoded paths are absolute
   - Understand why Docker volume mounts would make them work
   - Figure out the correct relative paths

   While sophisticated agents might deduce this, the failure is uniform across all model types, suggesting this is a non-obvious infrastructure difference rather than a solvable coding challenge.

## What This Fix Does NOT Do

- Does NOT automatically fix the paths (agents must modify the code themselves)
- Does NOT simplify the computational task
- Does NOT give hints about which dataset produces the expected answers
- Does NOT pre-compute any results
- Does NOT install any dependencies (agents must do this themselves)
- Does NOT change the questions being asked

The core challenge remains: understanding the codebase, modifying the path configuration, setting up the environment, running the model in test mode, and extracting the correct metrics from the output.
