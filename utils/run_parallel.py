import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
# You wanted index 0 to 49
TASK_IDS    = list(range(50))   

# ── TUNE THIS ─────────────────────────────────────────────────────────────────
# 2 is the safest for a MacBook (16GB RAM). 
# 4 is possible if you have 32GB+ RAM.
# Remember: Each worker loads its own copy of the GENRE model.
MAX_WORKERS = 2                 

def run_task(task_id: int):
    """Run perform_el.py for one task_id, stream output to a log file."""
    log_path = f"logs/el_task_{task_id}.log"
    
    # We use subprocess.run to isolate each task
    # This ensures if one task crashes, the others keep going
    with open(log_path, "w") as log:
        process = subprocess.run(
            [sys.executable, "perform_el.py", str(task_id)],
            stdout=log,
            stderr=subprocess.STDOUT, # Capture errors in the same log
        )
    return task_id, process.returncode, log_path

def main():
    print(f"🚀 Launching {len(TASK_IDS)} tasks (Indices 0-49)")
    print(f"   Max Workers: {MAX_WORKERS}")
    print(f"   Logs saved in: ./logs/\n")

    # ProcessPoolExecutor runs tasks in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the queue
        futures = {executor.submit(run_task, tid): tid for tid in TASK_IDS}

        for future in tqdm(as_completed(futures), total=len(TASK_IDS), desc="Overall Progress"):
            task_id, returncode, log_path = future.result()
            
            if returncode == 0:
                # print(f"  ✅ Task {task_id:>2} finished.")
                pass
            else:
                print(f"  ❌ Task {task_id:>2} FAILED (Exit Code {returncode}). Check {log_path}")

    print("\n🏁 All tasks completed.")

if __name__ == "__main__":
    from tqdm import tqdm # ensure you have tqdm installed: pip install tqdm
    main()