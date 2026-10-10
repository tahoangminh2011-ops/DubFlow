from __future__ import annotations
import shutil
from pathlib import Path

class CleanupManager:
    def __init__(self,job_dir:Path): self.job_dir=Path(job_dir)
    def cleanup(self):
        try: shutil.rmtree(self.job_dir,ignore_errors=True)
        except Exception: pass
