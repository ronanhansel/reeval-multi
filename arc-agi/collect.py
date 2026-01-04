import pandas as pd
from bs4 import BeautifulSoup
import sys

def extract_arc_matrix(html_file_path, output_csv_path):
    """
    Parses the ARC Prize HTML file and exports a binary response matrix.
    Rows: Models
    Cols: Task IDs
    Values: 1 (Pass), 0 (Fail), -1 (N/A)
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except FileNotFoundError:
        print(f"Error: File not found at {html_file_path}")
        return

    # --- 1. Extract Model Names ---
    # Model headers are in the first row of the <thead> and have colspan="2"
    header_row = soup.find('thead').find('tr')
    if not header_row:
        print("Error: Could not find table header.")
        return

    models = []
    headers = header_row.find_all('th')
    for th in headers:
        # Models span 2 columns (Status and Cost), metadata columns use rowspan
        if th.get('colspan') == '2':
            models.append(th.get_text(strip=True))

    print(f"Found {len(models)} models.")

    # --- 2. Extract Task Data ---
    rows = soup.find('tbody').find_all('tr')
    
    # Dictionary to store lists of scores for each model
    model_scores = {model: [] for model in models}
    task_ids = []

    for tr in rows:
        cells = tr.find_all('td')
        if not cells:
            continue

        # Task ID is in the 2nd column (index 1), usually inside an <a> tag
        task_id = cells[1].get_text(strip=True)
        task_ids.append(task_id)

        # Model data starts at index 4 (skipping Thumbnail, ID, Dataset, Human Panel)
        # Each model has 2 columns: Status and Cost. We only want Status (even indices starting at 4).
        current_col_idx = 4
        
        for model in models:
            if current_col_idx >= len(cells):
                # Handle cases where row might be malformed or incomplete
                model_scores[model].append(-1)
                continue

            status_cell = cells[current_col_idx]
            
            # Determine score based on badge class or text
            score = -1 # Default to N/A
            
            span = status_cell.find('span')
            if span:
                classes = span.get('class', [])
                if 'badge-pass' in classes:
                    score = 1
                elif 'badge-fail' in classes:
                    score = 0
                elif 'badge-na' in classes:
                    score = -1
            else:
                # Text fallback
                text = status_cell.get_text(strip=True).lower()
                if 'pass' in text:
                    score = 1
                elif 'fail' in text:
                    score = 0
            
            model_scores[model].append(score)
            
            # Move to the next model (skip the Cost column)
            current_col_idx += 2

    # --- 3. Create DataFrame and Save ---
    # Create DataFrame (Rows=Tasks, Cols=Models first)
    df = pd.DataFrame(model_scores, index=task_ids)
    
    # Transpose to get Rows=Models, Cols=Tasks
    df_transposed = df.T
    
    # Save to CSV
    df_transposed.to_csv(output_csv_path, index_label='modelname')
    print(f"Successfully saved matrix for {len(models)} models and {len(task_ids)} tasks to '{output_csv_path}'.")

# --- Usage ---
# Replace with your actual file path if different
input_file = "ARC Prize - Explore All Tasks.html"
output_file = "arc_prize_2_response_matrix.csv"

if __name__ == "__main__":
    extract_arc_matrix(input_file, output_file)