import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
load_dotenv() # load the .env file
os.makedirs(os.getenv("DATA_PATH"), exist_ok=True) # create the dataset directory if does not exist
snapshot_download(repo_id=os.getenv("DATA_CARD"), repo_type="dataset", local_dir=os.getenv("DATA_PATH"))