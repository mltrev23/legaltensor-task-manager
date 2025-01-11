import faiss
import pickle
import numpy as np

class VectorDatabase:
    def __init__(self, dimension=384, metric=faiss.METRIC_L2):
        """
        Initialize the vector database with a given dimension and metric.
        Default is L2 distance with 384 dimensions.
        """
        self.dimension = dimension
        self.metric = metric
        self.index = faiss.IndexFlat(dimension, metric)
        self.metadata = []

    def add(self, embeddings, meta):
        """
        Add embeddings and their corresponding metadata to the database.

        :param embeddings: A numpy array of shape (N, dimension).
        :param meta: A list of metadata corresponding to the embeddings.
        """
        if len(embeddings) != len(meta):
            raise ValueError("Number of embeddings and metadata must match.")
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype='float32')
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings must have {self.dimension} dimensions.")
        self.index.add(embeddings)
        self.metadata.extend(meta)

    def search(self, query_embedding, k=5):
        """
        Search for the top-k nearest neighbors of a query embedding.

        :param query_embedding: A numpy array of shape (1, dimension).
        :param k: The number of nearest neighbors to retrieve.
        :return: A list of tuples (distance, metadata).
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype='float32')
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding must have {self.dimension} dimensions.")
        distances, indices = self.index.search(query_embedding, k)
        results = [
            (distances[0][i], self.metadata[indices[0][i]])
            for i in range(len(indices[0]))
            if indices[0][i] != -1
        ]
        return results

    def save(self, filepath):
        """
        Save the vector database to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'index': faiss.serialize_index(self.index), 'metadata': self.metadata}, f)
            
    def load_from_file(self, filepath):
        """
        Load the vector database from a file and update the current instance.

        :param filepath: The path to the saved vector database file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data['index'])
        self.metadata = data['metadata']

    @staticmethod
    def load(filepath):
        """
        Load the vector database from a file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        db = VectorDatabase()
        db.index = faiss.deserialize_index(data['index'])
        db.metadata = data['metadata']
        return db

    def size(self):
        """
        Get the number of items in the database.
        """
        return self.index.ntotal
