import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import pickle
import os

# 1. Feature Engineering / Mock Data Generation
def generate_mock_hiring_data(num_samples=10000):
    """
    Generates mock data for the 5 specified features.
    """
    np.random.seed(42)
    
    # Feature 1: Cosine similarity of BharatBERT embeddings (0 to 1)
    bharatbert_sim = np.random.uniform(0.3, 0.95, num_samples)
    
    # Feature 2: KarmGraph proximity score (0 to 1, higher is closer)
    karmgraph_prox = np.random.uniform(0.1, 0.9, num_samples)
    
    # Feature 3: LSTM Trajectory Score (0 to 100)
    lstm_trajectory = np.random.uniform(20, 95, num_samples)
    
    # Feature 4: Behavioral engagement count (integer, e.g., clicks, app opens)
    behavioral_eng = np.random.poisson(lam=15, size=num_samples)
    
    # Feature 5: Verification factor (Boolean/Binary: 1 for verified, 0 for unverified)
    verification_factor = np.random.choice([0, 1], p=[0.3, 0.7], size=num_samples)
    
    # Target: Relevance Probability (Binary for training: 1 for Hired/Relevant, 0 for Not)
    # Let's create a logical relationship so the model learns something
    logit = (
        (bharatbert_sim * 3) + 
        (karmgraph_prox * 2) + 
        (lstm_trajectory / 100 * 1.5) + 
        (behavioral_eng / 30) + 
        (verification_factor * 1.2) - 4
    )
    prob = 1 / (1 + np.exp(-logit))
    target = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'bharatbert_sim': bharatbert_sim,
        'karmgraph_prox': karmgraph_prox,
        'lstm_trajectory': lstm_trajectory,
        'behavioral_eng': behavioral_eng,
        'verification_factor': verification_factor,
        'is_relevant': target
    })
    
    return df

# 2. LightGBM Training Pipeline
def train_lightgbm_combiner():
    print("Generating mock historical hiring data...")
    df = generate_mock_hiring_data()
    
    features = ['bharatbert_sim', 'karmgraph_prox', 'lstm_trajectory', 'behavioral_eng', 'verification_factor']
    X = df[features]
    y = df['is_relevant']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing LightGBM model...")
    # We use LGBMClassifier to output probability for 'Relevance Probability'
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    # Evaluation
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    loss = log_loss(y_test, preds_proba)
    print(f"\nModel Evaluation - AUC: {auc:.4f}, LogLoss: {loss:.4f}")
    
    # 3. Output Feature Importance Plot
    print("Generating feature importance plot...")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, importance_type='gain', max_num_features=5, title='Feature Importance (Gain)')
    plt.tight_layout()
    plot_path = 'feature_importance.png'
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")
    
    # 4. Save the model to .pkl
    model_path = 'combiner_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_lightgbm_combiner()
