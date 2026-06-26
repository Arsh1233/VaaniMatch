import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from collections import defaultdict

# 1. Knowledge Graph Construction (Mock data generation)
def construct_karmgraph():
    print("Constructing KarmGraph...")
    # Nodes
    skills = ['Python', 'Java', 'SQL', 'Machine Learning', 'React']
    companies = ['TCS', 'Infosys', 'Google', 'Amazon', 'StartupX']
    job_titles = ['Software Engineer', 'Data Scientist', 'Frontend Developer', 'Backend Developer']
    
    all_nodes = skills + companies + job_titles
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    
    # Edges (co-occurrence based on mock PMI)
    # E.g., Skill <-> Job Title, Company <-> Job Title
    edges_src = []
    edges_dst = []
    edge_weights = []
    
    # Generate mock edges with fake PMI weights
    for _ in range(50):
        src = random.choice(all_nodes)
        dst = random.choice(all_nodes)
        if src != dst:
            edges_src.append(node_to_id[src])
            edges_dst.append(node_to_id[dst])
            # Fake PMI weight
            edge_weights.append(random.uniform(0.1, 5.0))
            
    # Create DGL Graph
    g = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=len(all_nodes))
    g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
    
    # Initialize mock features for nodes (e.g., TF-IDF or random initialization)
    feature_dim = 64
    g.ndata['feat'] = torch.randn((g.num_nodes(), feature_dim))
    
    print(f"Graph constructed: {g.num_nodes()} nodes, {g.num_edges()} edges.")
    return g, node_to_id

# 2. GraphSAGE + GAT Architecture
class KarmGraphSAGE_GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=4):
        super(KarmGraphSAGE_GAT, self).__init__()
        # We combine SAGE and GAT ideas. Here we'll use a multi-head GAT layer 
        # which effectively aggregates neighborhood (like SAGE) but with attention.
        
        # Layer 1: GATConv
        self.gat1 = dglnn.GATConv(in_feats, hidden_feats, num_heads=num_heads, allow_zero_in_degree=True)
        # Layer 2: GATConv mapping to final output dimension
        self.gat2 = dglnn.GATConv(hidden_feats * num_heads, out_feats, num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        # First layer
        h = self.gat1(g, in_feat)
        h = h.view(h.shape[0], -1) # Flatten multi-head output
        h = F.elu(h)
        
        # Second layer
        h = self.gat2(g, h)
        h = h.squeeze(1) # Final output
        return h

# 3. Pipeline Execution
def run_pipeline():
    g, node_to_id = construct_karmgraph()
    
    # Model configuration
    in_feats = g.ndata['feat'].shape[1]
    hidden_feats = 64
    out_feats = 128 # Target output dimension
    
    model = KarmGraphSAGE_GAT(in_feats, hidden_feats, out_feats)
    
    # Mock Forward Pass
    model.eval()
    with torch.no_grad():
        embeddings = model(g, g.ndata['feat'])
        
    print(f"Generated embeddings shape: {embeddings.shape} (Expected: NumNodes x 128)")
    
    # 4. Save to .pt file
    output_path = 'karmgraph_embeddings.pt'
    torch.save({
        'embeddings': embeddings,
        'node_to_id': node_to_id
    }, output_path)
    print(f"Saved KarmGraph embeddings to {output_path}")

if __name__ == "__main__":
    run_pipeline()
