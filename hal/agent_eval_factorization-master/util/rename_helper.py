'''
Helper functions for renaming columns and standardizing data formats.
'''
import re
import pandas as pd

def rename_labels(df):
    rename_dict = {
        'selfcorrection.label': 'Self-Correction',
        'tooluse.label': 'Tool Use',
        'environmentalbarrier.label': 'Environmental Barrier',
        'verification.label': 'Verification',
        'instructionfollowing.label': 'Instruction Following',
        'binary_success_rate': 'Binary Success Rate'
    }
    return df.rename(columns=rename_dict)

def standardize_task_success_column(df, benchmark_name):
    '''
    Standardize the task success column to a binary 'success' column
    across different rubric datasets.
    '''
    if benchmark_name == "assistantbench":
        df = df.rename(columns={'eval_answer': 'binary_success_rate' })
    elif benchmark_name in ["corebench", "scicode", "taubench"]:
        df = df.rename(columns={'eval_is_successful': 'binary_success_rate'}) 
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")
    return df

def clean_model_name(model_name):
    """Standardize model names"""
    model_name = model_name.split('/')[-1]

    # Remove date patterns
    cleaned = re.sub(r'-\d{4}-\d{2}-\d{2}', '', model_name)
    
    # # Standardize specific model names
    # if cleaned.startswith('o3_medium'):
    #     cleaned = cleaned.replace('o3_medium', 'o3')
    return cleaned

def clean_rubric_name(rubric_name):
    rubric_name_map = {
        "selfcorrection.label": "Selfcorrection",
        "tooluse.label": "Tooluse",
        "environmentalbarrier.label": "Environmentalbarrier",
        "verification.label": "Verification",
        "instructionfollowing.label": "Instructionfollowing"
    }
    return rubric_name_map.get(rubric_name, rubric_name.replace('.label', ''))