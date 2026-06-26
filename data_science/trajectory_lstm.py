import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Architecture: LSTM with Attention
class TrajectoryAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TrajectoryAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Embedding layer for integer-encoded job roles
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Attention layer components
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Fully connected output layers to predict score (0-100)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        # Calculate attention scores
        attn_scores = self.attention_weights(lstm_out) # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Context vector (weighted sum of LSTM outputs)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        
        # Final prediction
        out = F.relu(self.fc1(context_vector))
        out = torch.sigmoid(self.fc2(out)) * 100 # Scale to 0-100
        
        return out.squeeze(1), attn_weights

# 2. Custom Loss Function
class StagnationPenaltyLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), penalty_weight=0.1):
        super(StagnationPenaltyLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets, job_sequences):
        """
        predictions: (batch_size) predicted scores
        targets: (batch_size) actual scores
        job_sequences: (batch_size, seq_len) original role sequences
        """
        mse = self.base_loss(predictions, targets)
        
        # Calculate stagnation penalty
        # Stagnation is defined here as contiguous identical roles.
        # We penalize sequences that have long runs of the same role integer.
        batch_penalty = 0
        for seq in job_sequences:
            stagnation_score = 0
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1] and seq[i] != 0: # Ignore padding if 0
                    current_run += 1
                else:
                    if current_run > 2: # Penalize if staying in same role for >2 time steps
                        stagnation_score += (current_run - 2) * 10 
                    current_run = 1
            if current_run > 2:
                stagnation_score += (current_run - 2) * 10
                
            batch_penalty += stagnation_score
            
        avg_penalty = batch_penalty / job_sequences.size(0)
        
        total_loss = mse + (self.penalty_weight * avg_penalty)
        return total_loss

# 3. Inference Code
def run_inference():
    # Mock parameters
    VOCAB_SIZE = 50 # Total unique job roles
    EMBED_DIM = 16
    HIDDEN_DIM = 32
    
    # Initialize model
    model = TrajectoryAttentionLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    model.eval()
    
    # Candidate's experience list (encoded as integers)
    # Example: [Junior Dev, Junior Dev, Mid Dev, Senior Dev, Lead]
    # Let's say: Junior=1, Mid=2, Senior=3, Lead=4
    candidate_experience = [1, 1, 2, 3, 4]
    
    # Stagnant example: [Junior Dev, Junior Dev, Junior Dev, Junior Dev]
    stagnant_experience = [1, 1, 1, 1]
    
    print("Running Inference...")
    with torch.no_grad():
        # Prepare inputs
        input_tensor_1 = torch.tensor([candidate_experience], dtype=torch.long)
        input_tensor_2 = torch.tensor([stagnant_experience], dtype=torch.long)
        
        # Get predictions
        score_1, attn_1 = model(input_tensor_1)
        score_2, attn_2 = model(input_tensor_2)
        
        print(f"Candidate 1 (Fast promotion): Experience={candidate_experience}")
        print(f"-> Predicted Trajectory Score: {score_1.item():.2f}")
        
        print(f"\nCandidate 2 (Stagnant): Experience={stagnant_experience}")
        print(f"-> Predicted Trajectory Score: {score_2.item():.2f}")

if __name__ == "__main__":
    run_inference()
