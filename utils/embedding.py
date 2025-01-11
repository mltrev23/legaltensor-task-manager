from sentence_transformers import SentenceTransformer
import torch

class TextToEmbedding:
    def __init__(self):
        # Use the Sentence-BERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, texts):
        # Generate embeddings for the texts
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().numpy()  # Convert embeddings to numpy array for FAISS compatibility