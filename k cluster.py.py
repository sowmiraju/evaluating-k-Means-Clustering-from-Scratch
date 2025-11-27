import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class KMeansScratch:
    """
    K-Means Clustering implementation from scratch using only NumPy
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        Initialize K-Means parameters
        
        Parameters:
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def initialize_centroids(self, X):
        """
        Initialize centroids using random selection from data points
        """
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        return self.centroids
    
    def assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        
        # Calculate distance from each point to each centroid
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        # Assign to nearest centroid
        self.labels = np.argmin(distances, axis=1)
        return self.labels
    
    def update_centroids(self, X):
        """
        Update centroids as the mean of assigned points
        """
        new_centroids = np.zeros_like(self.centroids)
        
        for i in range(self.k):
            # Get points assigned to cluster i
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # If no points assigned, reinitialize centroid
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
        
        return new_centroids
    
    def compute_inertia(self, X):
        """
        Compute within-cluster sum of squares (inertia)
        """
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.linalg.norm(cluster_points - self.centroids[i], axis=1) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means to the data
        """
        # Initialize centroids
        self.initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Assignment step
            self.assign_clusters(X)
            
            # Update step
            self.centroids = self.update_centroids(X)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            
            if centroid_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Compute final inertia
        self.inertia_ = self.compute_inertia(X)
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)

def generate_synthetic_dataset(n_samples=500, centers=4, cluster_std=0.8, random_state=42):
    """
    Generate synthetic dataset with clear cluster structure
    """
    X, y_true = make_blobs(
        n_samples=n_samples, 
        centers=centers, 
        cluster_std=cluster_std,
        random_state=random_state,
        n_features=2
    )
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, X.shape)
    X += noise
    
    return X, y_true

def evaluate_k_means(X, k_range=range(2, 8)):
    """
    Evaluate K-Means for different values of K
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"Evaluating K={k}...")
        
        # Initialize and fit K-Means
        kmeans = KMeansScratch(k=k)
        kmeans.fit(X)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(X, kmeans.labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette_avg)
        
        print(f"K={k}: Inertia = {inertia:.2f}, Silhouette Score = {silhouette_avg:.3f}")
    
    return inertias, silhouette_scores, k_range

def plot_results(X, y_pred, centroids, k, inertias, silhouette_scores, k_range):
    """
    Create comprehensive visualization of results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Clustering results
    scatter = axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    axes[0, 0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    axes[0, 0].set_title(f'K-Means Clustering (K={k})')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].legend()
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Plot 2: Elbow curve
    axes[0, 1].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters (K)')
    axes[0, 1].set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    axes[0, 1].set_title('Elbow Method for Optimal K')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Silhouette scores
    axes[1, 0].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Clusters (K)')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Silhouette Analysis for Optimal K')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cluster characteristics
    unique_labels = np.unique(y_pred)
    cluster_sizes = [np.sum(y_pred == label) for label in unique_labels]
    
    bars = axes[1, 1].bar(unique_labels, cluster_sizes, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
    axes[1, 1].set_xlabel('Cluster Label')
    axes[1, 1].set_ylabel('Number of Points')
    axes[1, 1].set_title('Cluster Sizes Distribution')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def analyze_cluster_characteristics(X, y_pred, centroids, k):
    """
    Analyze and print cluster characteristics
    """
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    for cluster_id in range(k):
        cluster_points = X[y_pred == cluster_id]
        cluster_size = len(cluster_points)
        centroid = centroids[cluster_id]
        
        if cluster_size > 0:
            # Calculate statistics
            mean_features = cluster_points.mean(axis=0)
            std_features = cluster_points.std(axis=0)
            cluster_radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            
            print(f"\nCluster {cluster_id}:")
            print(f"  - Size: {cluster_size} points ({cluster_size/len(X)*100:.1f}% of data)")
            print(f"  - Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
            print(f"  - Mean features: [{mean_features[0]:.3f}, {mean_features[1]:.3f}]")
            print(f"  - Std features: [{std_features[0]:.3f}, {std_features[1]:.3f}]")
            print(f"  - Maximum radius: {cluster_radius:.3f}")
        else:
            print(f"\nCluster {cluster_id}: Empty cluster")

def main():
    """
    Main function to run the complete K-Means analysis
    """
    print("K-MEANS CLUSTERING FROM SCRATCH")
    print("="*50)
    
    # Step 1: Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X, y_true = generate_synthetic_dataset(n_samples=400, centers=4, cluster_std=0.9)
    print(f"Dataset shape: {X.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # Step 2: Evaluate different K values
    print("\n2. Evaluating different K values...")
    k_range = range(2, 8)
    inertias, silhouette_scores, k_range = evaluate_k_means(X, k_range)
    
    # Step 3: Find optimal K
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal K based on silhouette score: {optimal_k}")
    
    # Step 4: Run K-Means with optimal K
    print(f"\n3. Running K-Means with optimal K={optimal_k}...")
    kmeans_optimal = KMeansScratch(k=optimal_k)
    kmeans_optimal.fit(X)
    y_pred = kmeans_optimal.labels
    
    # Step 5: Visualize results
    print("\n4. Generating visualizations...")
    plot_results(X, y_pred, kmeans_optimal.centroids, optimal_k, inertias, silhouette_scores, k_range)
    
    # Step 6: Analyze cluster characteristics
    analyze_cluster_characteristics(X, y_pred, kmeans_optimal.centroids, optimal_k)
    
    # Step 7: Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Optimal number of clusters (K): {optimal_k}")
    print(f"Final inertia (within-cluster sum of squares): {kmeans_optimal.inertia_:.2f}")
    print(f"Silhouette score: {silhouette_score(X, y_pred):.3f}")
    print(f"Number of iterations run: {kmeans_optimal.max_iters}")
    
    # Additional analysis: Compare with true labels (if available)
    if 'y_true' in locals():
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"Adjusted Rand Index (vs true labels): {ari:.3f}")

if __name__ == "__main__":
    main()
