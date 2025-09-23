import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
def get_reference_composite_eigenvector(a_df: pd.DataFrame, use_PCA=False) -> np.ndarray:
    """
    Calculates the eigenvector for the reference composite from a matrix of 
    multidimensional discrimination parameters.

    This vector corresponds to the largest eigenvalue of the a'a matrix and 
    represents the direction of the dominant unidimensional scale in the 
    multidimensional space.

    Args:
        a_df: A pandas DataFrame with shape (n_items, n_dimensions) 
              containing the 'a' parameters.

    Returns:
        A NumPy array representing the eigenvector of the reference composite.
    """
    # Convert the DataFrame to a NumPy array for matrix operations
    a_matrix = a_df.to_numpy()

    if use_PCA:
        pca = PCA(n_components=1)
        pc1_scores = pca.fit_transform(a_matrix)
        pc1_vec = pca.components_[0]
        pc1_unit = pc1_vec / (np.linalg.norm(pc1_vec) + 1e-12)
        return pc1_unit
    
    # Calculate the a-transpose-a matrix (a'a)
    a_t_a = a_matrix.T @ a_matrix
    
    # Calculate the eigenvalues and eigenvectors of the a'a matrix
    eigenvalues, eigenvectors = np.linalg.eig(a_t_a)
    
    # Find the index of the largest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)
    
    # Select the corresponding eigenvector
    reference_composite_vector = eigenvectors[:, max_eigenvalue_index]
    
    return reference_composite_vector

def orthogonalize_vector(vector_to_purify, reference_vector):
    """
    Purifies a vector by removing its projection onto a reference vector.
    This is a single step of the Gram-Schmidt process.

    Args:
        vector_to_purify (np.ndarray): The vector you want to purify (e.g., v_math).
        reference_vector (np.ndarray): The vector representing the component to remove (e.g., v_div).

    Returns:
        np.ndarray: The purified, orthogonal vector, normalized to unit length.
    """
    # Ensure vectors are normalized to unit length to start
    vector_to_purify = vector_to_purify / np.linalg.norm(vector_to_purify)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    # 1. Calculate the projection of vector_to_purify onto reference_vector
    dot_product = np.dot(vector_to_purify, reference_vector)
    projection = dot_product * reference_vector
    
    # 2. Subtract the projection to get the orthogonal component
    purified_vector = vector_to_purify - projection
    
    # 3. Normalize the final vector to have a length of 1
    purified_vector_normalized = purified_vector / np.linalg.norm(purified_vector)
    
    return purified_vector_normalized

def compute_scalar_ability(theta_df, a_df, b_series, item_ids, v_unit=None, adjust_difficulty=True, return_v_unit=False):
	"""
	Compute a math-specific ability score from MIRT factors, discriminations, and difficulties.
	
	Args:
		theta_df (pd.DataFrame): Factor scores per test taker (rows = test takers, cols = factors).
		a_df (pd.DataFrame): Discrimination parameters (rows = items, cols = factors).
		b_series (pd.Series): Difficulty parameters (indexed same as a_df rows).
		item_ids (list): List of item IDs (subset of a_df rows) that are math items.
		adjust_difficulty (bool): Whether to weight discriminations by difficulty.
	
	Returns:
		pd.Series: Math ability scores for each test taker.
	"""
	if v_unit is None:
		# 1. Extract discrimination vectors for math items
		A_math = a_df.loc[item_ids].copy()
		b_math = b_series.loc[item_ids]

		# 2. Optionally adjust by difficulty
		if adjust_difficulty:
			weights = 1 / (1 + np.abs(b_math.values))
			A_math = A_math.multiply(weights, axis=0)

		# 3. Compute math direction vector in factor space
		v_unit = A_math.mean(axis=0).values  # vector of length = num_factors
		v_unit = v_unit / np.linalg.norm(v_unit)  # normalize for stability
		print("Mean unit vector: ", v_unit)
		if return_v_unit:
			return v_unit

	else: 
		print("Used pre-computed direction vector.", v_unit)

	# 4. Project each test taker's theta onto v_unit
	theta_matrix = theta_df.values  # shape (n_test_takers, n_factors)
	mmlu_scores = theta_matrix.dot(v_unit)

	return pd.Series(mmlu_scores, index=theta_df.index, name="ability")

def process_reference_composite(resmat, theta_df, a_df, b_series, conv_dataset, use_PCA=False):
  items_mask = resmat.columns.get_level_values('scenario').isin(conv_dataset)

  scenario_questions = resmat.columns[items_mask].get_level_values('input.text').to_list()
  inverse_mask = ~items_mask
  inverse_questions = resmat.columns[inverse_mask].get_level_values('input.text').to_list()

  v_current_div_unit = compute_scalar_ability(theta_df, a_df, b_series, inverse_questions, return_v_unit=True)

  v_current_unit = get_reference_composite_eigenvector(a_df.loc[scenario_questions], use_PCA=use_PCA)
  v_current_mean_unit = compute_scalar_ability(theta_df, a_df, b_series, scenario_questions, return_v_unit=True)

  # --- Run the Orthogonalization ---
  v_current_pure = orthogonalize_vector(v_current_unit, v_current_div_unit)
  v_current_pure_mean = orthogonalize_vector(v_current_mean_unit, v_current_div_unit)
  current_scores_pure= compute_scalar_ability(theta_df, a_df, b_series, scenario_questions, v_current_pure)
  current_scores_pure_mean = compute_scalar_ability(theta_df, a_df, b_series, scenario_questions, v_current_pure_mean)
  div_scores_pure= compute_scalar_ability(theta_df, a_df, b_series, inverse_questions, v_current_div_unit)
  conv_benchmark_mean = resmat[resmat.columns[resmat.columns.get_level_values('scenario').isin(conv_dataset)]].mean(axis=1)

  div_current_corr = current_scores_pure.corr(div_scores_pure, method='spearman')
  print('div_corr', div_current_corr)

  conv_current_corr = current_scores_pure.corr(conv_benchmark_mean, method='spearman')
  print('conv_corr', conv_current_corr)

  div_current_corr_mean = current_scores_pure_mean.corr(div_scores_pure, method='spearman')
  print('div_corr_mean', div_current_corr_mean)

  conv_current_corr_mean = current_scores_pure_mean.corr(conv_benchmark_mean, method='spearman')
  print('conv_corr_mean', conv_current_corr_mean)

  if conv_current_corr < 0:
    print('conv_corr is negative, flipping the vector & scores')
    v_current_pure = -v_current_pure
    current_scores_pure = -current_scores_pure
  if conv_current_corr_mean < 0:
    print('conv_corr_mean is negative, flipping the vector & scores')
    v_current_pure_mean = -v_current_pure_mean
    current_scores_pure_mean = -current_scores_pure_mean
  return {
    'v': v_current_pure,
    'scores': current_scores_pure,
    'div_scores': div_scores_pure,
    'conv_scores': conv_benchmark_mean,
    'div_corr': div_current_corr,
    'conv_corr': conv_current_corr,
    'v_mean': v_current_pure_mean,
    'scores_mean': current_scores_pure_mean,
    'div_scores_mean': div_scores_pure,
    'conv_scores_mean': conv_benchmark_mean,
    'div_corr_mean': div_current_corr_mean,
    'conv_corr_mean': conv_current_corr_mean
  }

if __name__ == "__main__":
  import pandas as pd

  from load_params import load_and_rotate

  resmat = pd.read_pickle("../data/resmat.pkl")

  theta, a, b = load_and_rotate(rotation=None)

  # Step 1: Get the final theta tensor into a NumPy array
  theta_abilities = theta
  # Step 2: Create a labeled pandas DataFrame
  # Use the model names from your original resmat for the index
  model_names = resmat.index
  factor_names = [f'F{i+1}' for i in range(theta.shape[1])]
  ability_df = pd.DataFrame(theta_abilities, index=model_names, columns=factor_names)

  models = resmat.index.to_list()
  questions = resmat.columns.get_level_values('input.text').to_list()

  theta_df = pd.DataFrame(theta, index=models)

  a_df = pd.DataFrame(a, index=questions)

  b_series = pd.Series(b, index=questions)
# mmlu
# civil_comments
# med_qa
# math_gsm
  results_medqa = process_reference_composite(resmat, theta_df, a_df, b_series, ['med_qa'])
  print(results_medqa['div_corr'])
  print(results_medqa['conv_corr'])
  print(results_medqa['div_corr_mean'])
  print(results_medqa['conv_corr_mean'])