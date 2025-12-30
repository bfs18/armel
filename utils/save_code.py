import time
import tarfile
import wandb
from pathlib import Path

def save_code(exp_name, save_dir, save_to_wandb=True):
    # 1. Determine filename
    # If exp_name is None, try to get it from wandb
    if exp_name is None and wandb.run is not None:
        exp_name = wandb.run.name
        
    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    filename = f'code-{exp_name}-{time_str}.tar.gz' if exp_name else f'code-{time_str}.tar.gz'
    target_path = Path(save_dir) / filename
    
    # Ensure directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 2. Create tarball with specific structure (preserving project root name)
    proj_dir = Path(__file__).resolve().parent.parent
    
    # Extensions to include
    extensions = {'.py', '.yaml', '.sh', '.md', '.json'}
    # Directories to exclude
    exclude_dirs = {'.git', '.idea', '__pycache__', 'wandb', 'outputs', 'venv', '.venv'}
    
    with tarfile.open(target_path, "w:gz") as tar:
        for file_path in proj_dir.rglob("*"):
            if not file_path.is_file(): continue
            
            try:
                rel_path = file_path.relative_to(proj_dir)
            except ValueError: continue
            
            # Skip excluded directories
            if any(p in exclude_dirs for p in rel_path.parts): continue
            
            if file_path.suffix in extensions:
                # Key Requirement: Use arcname relative to parent to include project dir name
                # e.g. 'voxcpm-wave/utils/save_code.py'
                arcname = file_path.relative_to(proj_dir.parent).as_posix()
                tar.add(file_path, arcname=arcname)

    # 3. Upload artifact to WandB (Original logic restored but cleaner)
    if save_to_wandb and wandb.run:
        code_artifact = wandb.Artifact(
            name=f"source-code-{wandb.run.id}",
            type="code-tarball",
            description="Source code and configuration files"
        )
        code_artifact.add_file(str(target_path), name=f"source_code-{wandb.run.name}.tar.gz")
        wandb.run.log_artifact(code_artifact)
        print(f"Code artifact uploaded to wandb run: {wandb.run.name}")
