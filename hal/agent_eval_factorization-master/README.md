## Setup
```
# follow requirement.txt in /hal-harness and hal-paper-analysis
git clone https://github.com/princeton-pli/hal-harness/tree/main/agents
git clone https://github.com/peterkirgis/hal-paper-analysis.git

```

## Building Trace Matrix

Load & decrypt traces from Hal Agent Evaluation Leaderboard:
```
python trace_download.py
hal-decrypt -D traces
```
Outputs will be automatically stored inside `./traces`

### Compile your trace files:
Move all the traces you are interested in viewing into a `<directory>`. Then run the following:

```
# Generate a summary of all the traces in the directory.
python compile_traces.py <directory> --summarize

# Build a task-level success/failure matrix
python compile_traces.py <directory> --benchmark --build_matrix 

# Plot a single benchmark matrix
python compile_traces.py <directory> --benchmark <benchmark_name> --build_matrix 
```

## Building Rubric Matrix
The dataset for the rubric is inside the hal-paper-analysis repo. Clone this repo to get access to the data.

```
git clone https://github.com/peterkirgis/hal-paper-analysis.git
```

The rubrics can be found inside of `qualitative/results/rubrics`. Each rubric criteria has it's corresponding csv file. The column `label`, is the flag determining if that rubric criteria was flagged. See `/hal-paper-analysis/qualitative/README.md` for further details.

### Compile HAL traces:

```
# TODO, add flags here to compile the dataset
python compile_rubric_results.py
```

## Compule log analysis traces:
Make sure you have access to the `hal-paper-analysis` repo.

```
python compile_rubric_results.py
```

This code will output a file called `all_benchmarks_merged.csv`. This will contain all the log analysis traces into a single csv file. This file can be used for further analysis.

For viewing/analyzing results:
* Visualize the response matrix: `python analyze_rubric.py <some directory>/all_benchmarks_merged.csv --plot_matrix_by_rubric`
* Build correlation matrix: `python correlation.py`



### Running your own docent analysis on Hal
1. Create a docent account and create a new api_key. Set `DOCENT_API_KEY` in your environment.
2. Run to upload all the hal traces to your docent account:
```
python hal-paper-analysis/qualitative/full_pipeline.py
```
3. Go to docent and run your rubric.
