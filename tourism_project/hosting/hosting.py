from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",    
    repo_id="ShaksML/tourism",        

    repo_type="space",                     
    path_in_repo="",                         
)
