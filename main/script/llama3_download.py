
import os
from huggingface_hub import hf_hub_download

repo_id = "snower/omost-dolphin-2.9-llama3-8b-Q4_K_M-GGUF"
filename = "omost-dolphin-2.9-llama3-8b-q4_k_m.gguf"

cur_dir = os.path.dirname(__file__)
local_dir = os.path.join(cur_dir, "..", "models")
file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

print(f"File has been downloaded to: {file_path}")