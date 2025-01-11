import os
import requests
import numpy as np
import bittensor as bt
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI

from .utils.vector_db import VectorDatabase
from .utils.embedding import TextToEmbedding

class TaskApproveRequest(BaseModel):
    readme_md: str
    train_tsv: str
    test_tsv: str
    prompt_txt: str

# Contribution API Server
class ContributionAPIServer:
    def __init__(self, vector_db_file_path = None, score_threshold=0.7, approval_rate_threshold=0.7):
        load_dotenv()

        self.vector_db = VectorDatabase()
        if vector_db_file_path:
            self.vector_db.load_from_file(vector_db_file_path)

        self.embedding = TextToEmbedding()
        self.score_threshold = score_threshold
        self.approval_rate_threshold = approval_rate_threshold
        self.app = FastAPI()
        self.subnet_pool_url = os.environ.get('SUBNET_POOL_API')

        self.vpermit_tao_limit = 4096
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/submit_task")
        async def submit_task(data: TaskApproveRequest):
            task_id = hash(data.readme_md + data.prompt_txt)
            metadata = {
                "readme_md": data.readme_md,
                "train_tsv": data.train_tsv,
                "test_tsv": data.test_tsv,
                "prompt_txt": data.prompt_txt
            }
            
            task_data = {
                "task_id": task_id,
                "metadata": metadata
            }
            validator_endpoints = self.get_validators()
            scores = []
            for validator in validator_endpoints:
                score = requests.post(f"{validator}/score_task", json=task_data).json()
                scores.append(score)
            
            approval_rate = sum([score > self.score_threshold for score in scores])
            if approval_rate < self.approval_rate_threshold:
                return {"message": "Task not approved", "details": f"approval_rate: {approval_rate}, threshold: {self.approval_rate_threshold}"}

            avg = sum(scores) / len(scores)
            if avg < self.score_threshold:
                return {"message": "Task not approved", "details": f"overall_score_average: {avg}, threshold: {self.score_threshold}"}
        
            embedding = np.array(self.embedding.embed(data.readme_md))
            self.vector_db.add(embedding, metadata)
            
            return {"message": "Task submitted successfully!", "task_id": task_id}

    def get_validators(self):
        metagraph = bt.metagraph(205, 'test')
        avail_uids = list(metagraph.n.items())
        validator_uids = [metagraph.S[uid] > self.vpermit_tao_limit for uid in avail_uids]
        return requests.post(self.subnet_pool_url + '/healthy-endpoints', json = {'uids': validator_uids}).json()
