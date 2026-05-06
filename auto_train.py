import subprocess
import time


def run_training(script_path, max_repeats=6, rest_time=5 * 60):
    """
    Automatically run train_supervision.py
    :param script_path: Path to the training script
    :param max_repeats: Total number of training runs
    :param rest_time: Rest time after each training run (seconds)
    """
    for i in range(max_repeats):
        print(f"[INFO] Training run {i + 1} started...")

        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8")
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"[INFO] Training run {i + 1} completed!")
        else:
            print(f"[ERROR] Training run {i + 1} failed, error details below:")
            print(stderr)
            # break  # Stop the loop when an error occurs

        if i < max_repeats - 1:
            print(f"[INFO] Resting for {rest_time // 60} minutes before continuing...")
            time.sleep(rest_time)

    print("[INFO] All training tasks completed!")


if __name__ == "__main__":
    script_path = "train_supervision.py"  # Ensure this path is correct
    run_training(script_path)