import pickle
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from weaviate import Client

class VectorDatabase:
    def __init__(self, weaviate_url: str):
        """
        Initialize TaskManager with a connection to a Weaviate vector database.

        :param weaviate_url: URL of the Weaviate instance
        """
        self.client = Client(weaviate_url)

        # Define the schema for task objects in the vector database
        if not self.client.schema.exists('Task'):
            self.client.schema.create({
                "classes": [
                    {
                        "class": "Task",
                        "properties": [
                            {
                                "name": "data",
                                "dataType": ["text"]
                            },
                            {
                                "name": "embedding",
                                "dataType": ["number[]"]
                            }
                        ]
                    }
                ]
            })

    def add(self, data: Dict[str, str], embedding: np.ndarray):
        """
        Add a new data entry along with its embedding to the vector database.

        :param data: A dictionary containing task data (e.g., readme, prompt, etc.)
        :param embedding: The embedding vector representing the task
        """
        self.client.data_object.create({
            "data": data,
            "embedding": embedding.tolist()
        }, "Task")

    def get_all(self) -> List[Tuple[np.ndarray, Dict[str, str]]]:
        """
        Retrieve all entries from the vector database.

        :return: A list of tuples where each tuple contains an embedding and its corresponding data
        """
        results = self.client.data_object.get(class_name="Task")
        tasks = []

        for obj in results["objects"]:
            embedding = np.array(obj["vector"])
            data = obj["properties"]["data"]
            tasks.append((embedding, data))

        return tasks

    def save(self, file_path: str):
        """
        Save the current state of the vector database to a file (Weaviate doesn't support direct export).

        :param file_path: Path to the file where the database will be saved
        """
        all_data = self.get_all()
        with open(file_path, 'wb') as file:
            pickle.dump(all_data, file)

    def load_from_file(self, file_path: str):
        """
        Load the database from a file and populate the vector database.

        :param file_path: Path to the file from which the database will be loaded
        """
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)

        for embedding, data in all_data:
            self.add(data, embedding)

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, str]]:
        """
        Search the vector database for the closest embedding to the query embedding.

        :param query_embedding: The embedding vector to search for
        :param top_k: Number of top similar results to retrieve
        :return: A list of data dictionaries corresponding to the closest embeddings
        """
        query_vector = query_embedding.tolist()
        results = self.client.query.get("Task", ["data"]).with_near_vector({"vector": query_vector}).with_limit(top_k).do()

        return [obj["properties"]["data"] for obj in results["data"]["Get"]["Task"]]
