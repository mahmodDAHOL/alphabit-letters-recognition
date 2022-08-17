import subprocess
from pathlib import Path


virtual_environment_name = "venv"

training = False
testing_real_time = True

project_path = Path("main.py").parent.absolute().parent
project_folders = ["models", "results/",
                   "results/cnn_training_result"]
virtual_environment_path = project_path.joinpath(
    f"src/{virtual_environment_name}/Scripts")

for project_folder in project_folders:
    if not project_path.joinpath(project_folder).exists():
        print("[INFO] create required folders.....")
        project_path.joinpath(project_folder).mkdir()

activate_env_path = virtual_environment_path.joinpath("activate.bat")
python_path = virtual_environment_path.joinpath("python.exe")
pip_path = virtual_environment_path.joinpath("pip.exe")
requirements = project_path.joinpath("requirements.txt")

if not activate_env_path.exists():
    print("[INFO] create virtual environment.....")
    subprocess.run(
        ['python', '-m', 'venv', virtual_environment_name])

print("[INFO] activate the virtual environment....")
subprocess.run([activate_env_path])
print("[INFO] install all packages......")
subprocess.run([pip_path, 'install', '-r', requirements])

if training:
    print("[INFO] run training.py......")
    subprocess.run(["python", "training.py"])


if testing_real_time:
    print("[INFO] run test.py......")
    subprocess.run(["python", "test.py"])
