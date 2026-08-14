import sys
import os

repo_dir = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_dir)
sys.path.insert(0, repo_dir)

import launch
launch.prepare_environment()
print("Finished Forge Neo preinstall")
