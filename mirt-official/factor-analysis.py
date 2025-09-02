import pandas as pd
from factor_analyzer import FactorAnalyzer
resmat = pd.read_pickle("../data/resmat.pkl").fillna(0)

# --- Run Parallel Analysis Correctly ---
# Create a FactorAnalyzer object. We can check up to 25 factors.
# The library handles all the simulation and comparison internally.
fa = FactorAnalyzer(n_factors=25, rotation=None, )
fa.fit(resmat)

# Get the raw eigenvalues from your actual data and the simulated random data
real_eigenvalues, random_eigenvalues = fa.get_eigenvalues()

# --- Find the correct number of factors ---
n_factors_to_retain = sum(real_eigenvalues > random_eigenvalues)

print("\n--- Parallel Analysis Results (Corrected Method) ---")
print(f"Based on the comparison of raw eigenvalues, you should retain: {n_factors_to_retain} factors.")

# Optional: Plot the results to create a proper scree plot
import matplotlib.pyplot as plt

plt.plot(real_eigenvalues, '--bo', label='Actual Data Eigenvalues')
plt.plot(random_eigenvalues, '--rx', label='Random Data Eigenvalues')
plt.title('Parallel Analysis Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.legend()
plt.savefig('./parallel_analysis_scree_plot.png')
plt.show()