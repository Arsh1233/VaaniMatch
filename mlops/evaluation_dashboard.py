import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CELL 1: Setup & Mock Data Generation
# ==========================================
def generate_mock_evaluation_data(num_jds=500, pool_size=1000):
    """
    Generates mock search results for 500 JDs against a pool.
    Returns dictionaries containing the rank positions of the *true* positive candidates.
    """
    np.random.seed(42)
    
    # We assume for each JD, there is 1 'perfect' candidate we want to retrieve.
    # Keyword Baseline: The perfect candidate is often ranked lower (or not found).
    # VaaniMatch: The perfect candidate is ranked higher on average.
    
    keyword_ranks = []
    vaanimatch_ranks = []
    
    for _ in range(num_jds):
        # Keyword might not find it (assign rank > pool_size), or find it randomly
        kw_rank = int(np.random.exponential(scale=200)) + 1
        if kw_rank > pool_size: kw_rank = pool_size + 1
        keyword_ranks.append(kw_rank)
        
        # VaaniMatch uses semantics, so rank is much better
        vm_rank = int(np.random.exponential(scale=15)) + 1
        vaanimatch_ranks.append(vm_rank)
        
    return np.array(keyword_ranks), np.array(vaanimatch_ranks)

# ==========================================
# CELL 2: Evaluation Metrics Calculation
# ==========================================
def calc_precision_at_k(ranks, k=10):
    # Precision@K: 1 if the true positive is in top K, else 0. (Averaged across all queries)
    hits = (ranks <= k).sum()
    return hits / len(ranks)

def calc_recall_at_k(ranks, k=50):
    # For a single true positive, Recall@K is the same as Precision@K just evaluated at a wider net.
    hits = (ranks <= k).sum()
    return hits / len(ranks)

def calc_mrr(ranks):
    # MRR: Mean Reciprocal Rank (1/rank)
    # If rank is outside pool, reciprocal is 0
    reciprocals = [1.0 / r if r <= 1000 else 0.0 for r in ranks]
    return np.mean(reciprocals)

def calc_ndcg(ranks, k=100):
    # NDCG: Normalized Discounted Cumulative Gain
    # For binary relevance of a single item, DCG = 1 / log2(rank + 1) if rank <= k
    # IDCG (Ideal) is 1 / log2(1 + 1) = 1. So NDCG = DCG.
    ndcg_scores = []
    for r in ranks:
        if r <= k:
            ndcg_scores.append(1.0 / np.log2(r + 1))
        else:
            ndcg_scores.append(0.0)
    return np.mean(ndcg_scores)

# ==========================================
# CELL 3: Generate Metrics & Plot
# ==========================================
def main():
    print("Generating test dataset of 500 labeled JDs...")
    kw_ranks, vm_ranks = generate_mock_evaluation_data(num_jds=500)
    
    # Calculate metrics
    metrics = {
        'Precision@10': (calc_precision_at_k(kw_ranks, 10), calc_precision_at_k(vm_ranks, 10)),
        'Recall@50': (calc_recall_at_k(kw_ranks, 50), calc_recall_at_k(vm_ranks, 50)),
        'MRR': (calc_mrr(kw_ranks), calc_mrr(vm_ranks)),
        'NDCG': (calc_ndcg(kw_ranks, 100), calc_ndcg(vm_ranks, 100))
    }
    
    # Prepare DataFrame for Seaborn
    data = []
    for metric, (kw_val, vm_val) in metrics.items():
        data.append({'Metric': metric, 'System': 'Keyword-Baseline', 'Score': kw_val})
        data.append({'Metric': metric, 'System': 'VaaniMatch', 'Score': vm_val})
        
    df = pd.DataFrame(data)
    
    # Plotting
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    
    # Create the bar chart
    ax = sns.barplot(
        data=df, 
        x='Metric', 
        y='Score', 
        hue='System',
        palette=["#e74c3c", "#2ecc71"] # Red for baseline, Green for VaaniMatch
    )
    
    plt.title('Retrieval Performance: VaaniMatch vs. Keyword-Baseline (500 JDs)', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel('Score (0 to 1)')
    plt.legend(title='System')
    
    # Add value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=10, color='black', 
                        xytext=(0, 5), textcoords='offset points')
            
    plt.tight_layout()
    
    output_img = "evaluation_dashboard.png"
    plt.savefig(output_img, dpi=300)
    print(f"Chart successfully saved to {output_img}")
    
    print("\n--- Summary of Metrics ---")
    for metric, (kw_val, vm_val) in metrics.items():
        print(f"{metric:>12}: Baseline = {kw_val:.4f} | VaaniMatch = {vm_val:.4f}")

if __name__ == "__main__":
    main()
