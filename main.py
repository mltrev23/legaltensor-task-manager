import os
import requests
import bittensor as bt
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI

class TaskApproveRequest(BaseModel):
    readme_md: str
    train_tsv: str
    test_tsv: str
    prompt_txt: str

# Contribution API Server
class ContributionAPIServer:
    def __init__(self, vector_db, threshold=0.7):
        load_dotenv()

        self.vector_db = vector_db
        self.threshold = threshold
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
            
            return {"message": "Task submitted successfully!", "task_id": task_id}

    def get_validators(self):
        metagraph = bt.metagraph(205, 'test')
        avail_uids = list(metagraph.n.items())
        validator_uids = [metagraph.S[uid] > self.vpermit_tao_limit for uid in avail_uids]
        return requests.post(self.subnet_pool_url + '/healthy-endpoints', json = {'uids': validator_uids}).json()

    def check_approval(self, task_id, avg_score):
        if avg_score >= self.threshold:
            print(f"Task {task_id} approved with average score {avg_score}.")
        else:
            print(f"Task {task_id} rejected with average score {avg_score}.")
