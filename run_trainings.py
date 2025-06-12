from datetime import datetime
import traceback
from utils import get_runpod_config, get_output_dir, set_hf_home_path
import os
import subprocess
from model_intervl import run_model_training


def cleanup_env():
    try:
        # Get the current date and time
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        outfile = os.path.join(get_output_dir(), 'exithistory.txt')
        with open(outfile, 'a') as file:
            file.write(f"{formatted_time} done\n")
    except Exception as e:
        print(e)
    runpod_id, runpod_terminate_on_exit = get_runpod_config()
    if runpod_id is not None and runpod_terminate_on_exit:
        print(f'Terminating runpod {runpod_id}')
        result = subprocess.run(["runpodctl", "remove", "pod", runpod_id])
        if result.returncode == 0:
            print("Pod successfully removed")
        else:
            print(f"Error removing pod: {result.stderr}")

if __name__ == "__main__":
    try:
        set_hf_home_path()
        run_model_training()
    except Exception as e:
        traceback.print_exc()
    finally:
        cleanup_env()
