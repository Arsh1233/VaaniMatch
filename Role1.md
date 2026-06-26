### Role 1: Data Scientist (NLP & Graph Modeling)
*Focus: Building BharatBERT, KarmGraph, and Trajectory models.*

**Prompt 1.1 (BharatBERT Fine-tuning):**
> "Write a Python script using PyTorch and HuggingFace Transformers to fine-tune a `distilbert-base-multilingual-cased` model on a synthetic dataset of 5,000 Indian job descriptions mixed with Hindi, Tamil, and English. Implement a contrastive loss function (InfoNCE) so that the model generates 256-dim embeddings where JD embeddings are pulled closer to relevant Resume embeddings. Include data augmentation by replacing English tech terms (e.g., 'Python') with their Hindi transliterations to simulate real-world Indian CVs."

**Prompt 1.2 (KarmGraph Construction):**
> "Develop a Knowledge Graph pipeline using Deep Graph Library (DGL) and NetworkX. Construct nodes for 'Skills', 'Companies', and 'Job Titles'. Generate weighted edges based on co-occurrence frequencies (PMI) from 100k historical resumes. Write a GraphSAGE implementation with GAT (Graph Attention) convolution layers that outputs 128-dim node embeddings. Provide the code to save these embeddings to a `.pt` file for downstream retrieval."

**Prompt 1.3 (Career Trajectory LSTM):**
> "Write an LSTM with Attention mechanism in PyTorch to model career progression. Input is a sequence of job roles (encoded as integers) over time. Output is a 'Trajectory Score' (0-100) representing promotion velocity and skill acquisition rate. The architecture should include a custom loss function that penalizes stagnation (long durations in the same role). Provide inference code to feed a candidate's experience list into this model."

**Prompt 1.4 (LightGBM Combiner):**
> "Create a training pipeline using LightGBM that takes five engineered features as input: (1) Cosine similarity of BharatBERT embeddings, (2) KarmGraph proximity score, (3) LSTM Trajectory Score, (4) Behavioral engagement count, and (5) Verification factor. Train this model on historical hiring data to predict a final 'Relevance Probability'. Output the feature importance plot and the `.pkl` model file."

---