"""
GPU-Accelerated UMAP Analysis for 14 Million Sample Dataset
Author: AI Assistant
Description: High-performance UMAP implementation with GPU acceleration for large-scale dimensionality reduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import warnings
import time
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def install_gpu_dependencies():
    """
    Install GPU-accelerated UMAP dependencies
    """
    print("Installing GPU dependencies...")
    print("For RAPIDS cuML (recommended for NVIDIA GPUs):")
    print("conda install -c rapidsai -c conda-forge cuml=23.12 python=3.10 cudatoolkit=11.8")
    print("\nFor CPU fallback:")
    print("pip install umap-learn[plot]")
    print("\nFor interactive plotting:")  
    print("pip install plotly bokeh")

def setup_umap_gpu():
    """
    Setup UMAP with GPU acceleration, with CPU fallback
    """
    try:
        # Try GPU-accelerated UMAP first (RAPIDS cuML)
        print("Attempting to load GPU-accelerated UMAP (cuML)...")
        from cuml.manifold import UMAP as cuUMAP
        print("✅ GPU-accelerated UMAP (cuML) loaded successfully!")
        return cuUMAP, "GPU"
    except ImportError:
        try:
            # Fallback to CPU UMAP
            print("GPU UMAP not available, falling back to CPU UMAP...")
            from umap import UMAP
            print("✅ CPU UMAP loaded successfully!")
            return UMAP, "CPU"
        except ImportError:
            print("❌ Neither GPU nor CPU UMAP available!")
            print("Please install: pip install umap-learn")
            return None, None

class UMAPAnalyzer:
    """
    High-performance UMAP analyzer with GPU acceleration for large datasets
    """
    
    def __init__(self, data_path="../data/resmat.pkl", result_path="../result/"):
        self.data_path = data_path
        self.result_path = result_path
        self.umap_class, self.device = setup_umap_gpu()
        self.scenario_factors = None
        self.scenario_labels = None
        self.umap_2d = None
        self.umap_3d = None
        
        # Ensure result directory exists
        os.makedirs(result_path, exist_ok=True)
        
    def load_data(self):
        """
        Load and prepare data for UMAP analysis
        """
        print("Loading data...")
        
        # Load scenario factors (equivalent to your a_for_z_numpy)
        try:
            self.scenario_factors = np.load(f"{self.result_path}scenario_factors.npy")
            print(f"Loaded existing scenario factors: {self.scenario_factors.shape}")
        except FileNotFoundError:
            print("Scenario factors not found, loading from MIRT model...")
            # If you haven't saved scenario_factors.npy yet, load from your MIRT model
            sys.path.append('/home/azureuser/cloudfiles/code/Users/manhductranvu/reeval-multi/mirt-official')
            from load_params import load_and_rotate
            _, self.scenario_factors, _ = load_and_rotate('./output/mirt_model_k19_auc89.pt')
            np.save(f"{self.result_path}scenario_factors.npy", self.scenario_factors)
            print(f"Loaded and saved scenario factors: {self.scenario_factors.shape}")
        
        # Load scenario labels
        resmat = pd.read_pickle(self.data_path)
        balanced_resmat = resmat.columns.to_frame()
        self.scenario_labels = balanced_resmat['scenario'].values
        
        print(f"Data loaded: {self.scenario_factors.shape[0]:,} samples, {self.scenario_factors.shape[1]} dimensions")
        return self
    
    def run_umap_2d(self, n_neighbors=30, min_dist=0.1, metric='euclidean', 
                    spread=1.0, random_state=42, verbose=True):
        """
        Run 2D UMAP with optimized parameters for large datasets
        """
        print(f"\n🚀 Running 2D UMAP on {self.device}...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        
        start_time = time.time()
        
        if self.device == "GPU":
            # GPU-optimized parameters
            umap_model = self.umap_class(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                spread=spread,
                random_state=random_state,
                verbose=verbose
            )
        else:
            # CPU parameters with performance optimizations
            umap_model = self.umap_class(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                spread=spread,
                random_state=random_state,
                verbose=verbose,
                n_jobs=-1,  # Use all CPU cores
                low_memory=False  # Use more memory for speed
            )
        
        self.umap_2d = umap_model.fit_transform(self.scenario_factors)
        
        end_time = time.time()
        print(f"✅ 2D UMAP completed in {end_time - start_time:.2f} seconds")
        print(f"Result shape: {self.umap_2d.shape}")
        
        # Save results
        np.save(f"{self.result_path}umap_2d.npy", self.umap_2d)
        print(f"Saved to {self.result_path}umap_2d.npy")
        
        return self.umap_2d
    
    def run_umap_3d(self, n_neighbors=30, min_dist=0.1, metric='euclidean', 
                    spread=1.0, random_state=42, verbose=True):
        """
        Run 3D UMAP for landscape visualization
        """
        print(f"\n🚀 Running 3D UMAP on {self.device}...")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        
        start_time = time.time()
        
        if self.device == "GPU":
            umap_model = self.umap_class(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                spread=spread,
                random_state=random_state,
                verbose=verbose
            )
        else:
            umap_model = self.umap_class(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                spread=spread,
                random_state=random_state,
                verbose=verbose,
                n_jobs=-1,
                low_memory=False
            )
        
        self.umap_3d = umap_model.fit_transform(self.scenario_factors)
        
        end_time = time.time()
        print(f"✅ 3D UMAP completed in {end_time - start_time:.2f} seconds")
        print(f"Result shape: {self.umap_3d.shape}")
        
        # Save results
        np.save(f"{self.result_path}umap_3d.npy", self.umap_3d)
        print(f"Saved to {self.result_path}umap_3d.npy")
        
        return self.umap_3d
    
    def plot_2d_highlighted(self, labels_to_highlight=['legal_support', 'math', 'gsm', 'lsat_qa'],
                          figsize=(16, 12), save_name="umap_2d_highlighted.pdf"):
        """
        Create 2D UMAP visualization with highlighted scenarios
        """
        if self.umap_2d is None:
            print("❌ Run 2D UMAP first!")
            return
        
        print("\n📊 Creating 2D UMAP visualization...")
        
        # Setup colors
        default_color = 'lightgrey'
        colors = plt.colormaps.get_cmap('tab10')
        color_map = {label: colors(i) for i, label in enumerate(labels_to_highlight)}
        
        plt.figure(figsize=figsize)
        
        # Plot non-highlighted points first
        non_highlighted_mask = ~np.isin(self.scenario_labels, labels_to_highlight)
        if np.any(non_highlighted_mask):
            plt.scatter(self.umap_2d[non_highlighted_mask, 0], 
                       self.umap_2d[non_highlighted_mask, 1], 
                       c=default_color, alpha=0.05, s=10, zorder=1)
        
        # Plot highlighted points
        for label in labels_to_highlight:
            mask = self.scenario_labels == label
            if np.any(mask):
                plt.scatter(self.umap_2d[mask, 0], self.umap_2d[mask, 1], 
                           c=color_map[label], alpha=0.8, s=20, zorder=2, label=label)
        
        plt.title('UMAP 2D Visualization of Test Scenarios', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=14)
        plt.ylabel('UMAP Dimension 2', fontsize=14)
        plt.legend(title='Highlighted Scenarios', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.result_path}{save_name}", dpi=300, bbox_inches='tight')
        print(f"Saved to {self.result_path}{save_name}")
        plt.show()
    
    def plot_3d_landscape(self, labels_to_highlight=['legal_support', 'math', 'gsm', 'lsat_qa'],
                         figsize=(16, 12), save_name="umap_3d_landscape.pdf"):
        """
        Create 3D UMAP landscape visualization
        """
        if self.umap_3d is None:
            print("❌ Run 3D UMAP first!")
            return
        
        print("\n📊 Creating 3D UMAP landscape visualization...")
        
        # Setup colors
        default_color = 'lightgrey'
        colors = plt.colormaps.get_cmap('tab10')
        color_map = {label: colors(i) for i, label in enumerate(labels_to_highlight)}
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot non-highlighted points
        non_highlighted_mask = ~np.isin(self.scenario_labels, labels_to_highlight)
        if np.any(non_highlighted_mask):
            ax.scatter(self.umap_3d[non_highlighted_mask, 0], 
                      self.umap_3d[non_highlighted_mask, 1], 
                      self.umap_3d[non_highlighted_mask, 2], 
                      c=default_color, alpha=0.3, s=8, depthshade=True)
        
        # Plot highlighted points
        handles = []
        for label in labels_to_highlight:
            mask = self.scenario_labels == label
            if np.any(mask):
                ax.scatter(self.umap_3d[mask, 0], self.umap_3d[mask, 1], 
                          self.umap_3d[mask, 2], c=color_map[label], 
                          alpha=0.8, s=20, depthshade=True, label=label, 
                          edgecolors='black', linewidth=0.3)
                
                handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                        markerfacecolor=color_map[label], markersize=10,
                                        markeredgecolor='black', markeredgewidth=0.3))
        
        # Styling
        ax.set_xlabel('UMAP Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12, labelpad=10) 
        ax.set_zlabel('UMAP Dimension 3', fontsize=12, labelpad=10)
        ax.set_title('3D UMAP Landscape of Test Scenarios', fontsize=14, pad=20)
        ax.view_init(elev=0, azim=0)
        ax.grid(True, alpha=0.3)
        
        # Pane styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        ax.legend(handles=handles, title='Scenarios', bbox_to_anchor=(1.15, 1), 
                  loc='upper left', fontsize=10, title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.result_path}{save_name}", bbox_inches='tight')
        print(f"Saved to {self.result_path}{save_name}")
        plt.show()
    
    def hyperparameter_search(self, param_grid=None):
        """
        Perform hyperparameter search for optimal UMAP parameters
        """
        if param_grid is None:
            param_grid = {
                'n_neighbors': [15, 30, 50, 100],
                'min_dist': [0.05, 0.1, 0.3, 0.5],
                'metric': ['euclidean', 'cosine', 'manhattan']
            }
        
        print("\n🔍 Starting hyperparameter search...")
        print("This will test different parameter combinations...")
        
        results = []
        total_combinations = len(param_grid['n_neighbors']) * len(param_grid['min_dist']) * len(param_grid['metric'])
        
        for i, n_neighbors in enumerate(param_grid['n_neighbors']):
            for j, min_dist in enumerate(param_grid['min_dist']):
                for k, metric in enumerate(param_grid['metric']):
                    combo_num = i * len(param_grid['min_dist']) * len(param_grid['metric']) + j * len(param_grid['metric']) + k + 1
                    print(f"Testing combination {combo_num}/{total_combinations}: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
                    
                    start_time = time.time()
                    
                    try:
                        if self.device == "GPU":
                            umap_model = self.umap_class(
                                n_components=2,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                metric=metric,
                                random_state=42,
                                verbose=False
                            )
                        else:
                            umap_model = self.umap_class(
                                n_components=2,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                metric=metric,
                                random_state=42,
                                verbose=False,
                                n_jobs=-1
                            )
                        
                        # Use a subset for hyperparameter search to save time
                        subset_size = min(10000, len(self.scenario_factors))
                        subset_indices = np.random.choice(len(self.scenario_factors), subset_size, replace=False)
                        subset_data = self.scenario_factors[subset_indices]
                        
                        embedding = umap_model.fit_transform(subset_data)
                        end_time = time.time()
                        
                        results.append({
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'metric': metric,
                            'time': end_time - start_time,
                            'embedding_std': np.std(embedding),
                            'embedding_range': np.ptp(embedding)
                        })
                        
                        print(f"  ✅ Completed in {end_time - start_time:.2f}s")
                        
                    except Exception as e:
                        print(f"  ❌ Failed: {e}")
                        continue
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.result_path}umap_hyperparameter_results.csv", index=False)
        print(f"\n📊 Hyperparameter search complete! Results saved to {self.result_path}umap_hyperparameter_results.csv")
        
        # Show best parameters based on embedding quality
        if len(results_df) > 0:
            best_params = results_df.loc[results_df['embedding_std'].idxmax()]
            print("\n🏆 Best parameters (highest embedding standard deviation):")
            print(f"n_neighbors: {best_params['n_neighbors']}")
            print(f"min_dist: {best_params['min_dist']}")
            print(f"metric: {best_params['metric']}")
            print(f"Time: {best_params['time']:.2f}s")
        
        return results_df


def main():
    """
    Main execution function
    """
    print("🚀 GPU-Accelerated UMAP Analysis for Large-Scale Data")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = UMAPAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    if analyzer.umap_class is None:
        print("❌ UMAP not available. Please install dependencies.")
        install_gpu_dependencies()
        return
    
    print(f"\n📊 Using {analyzer.device} acceleration")
    print(f"Data shape: {analyzer.scenario_factors.shape}")
    
    # # Run hyperparameter search (optional - comment out if you want to skip)
    # # print("\n🔍 Running hyperparameter search on subset...")
    # # analyzer.hyperparameter_search()
    
    # # Run 2D UMAP with optimized parameters for large datasets
    # print("\n🚀 Running 2D UMAP...")
    # analyzer.run_umap_2d(
    #     n_neighbors=50,      # Larger for smoother embeddings
    #     min_dist=0.1,        # Good balance of local/global structure
    #     metric='euclidean',   # Fast and effective
    #     spread=1.5,          # Slightly more spread for better separation
    #     verbose=True
    # )
    
    # # Create 2D visualization
    # analyzer.plot_2d_highlighted()
    
    # Run 3D UMAP
    print("\n🚀 Running 3D UMAP...")
    analyzer.run_umap_3d(
        n_neighbors=50,
        min_dist=0.1,
        metric='euclidean',
        spread=1.5,
        verbose=True
    )
    
    # Create 3D visualization
    analyzer.plot_3d_landscape()
    
    print("\n✅ Analysis complete!")
    print(f"Results saved in: {analyzer.result_path}")
    print("\nFiles created:")
    print("- umap_2d.npy: 2D UMAP embeddings")
    print("- umap_3d.npy: 3D UMAP embeddings") 
    print("- umap_2d_highlighted.pdf: 2D visualization")
    print("- umap_3d_landscape.pdf: 3D landscape visualization")


if __name__ == "__main__":
    main()
