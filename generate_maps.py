import subprocess
import sys

def main():
    for i in range(1000):
        subprocess.run([sys.executable, "-m", "tools.generate_maps","run=mapgen","eval.env=env/mettagrid/configs/mapgen", "+hardware=macbook", "cmd=generate_maps"])

if __name__ == "__main__":
    main()
