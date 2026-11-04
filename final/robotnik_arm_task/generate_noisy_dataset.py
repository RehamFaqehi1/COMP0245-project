"""Step 3: collect/resume noisy expert trajectories, export and plot results."""
import sys
sys.dont_write_bytecode = True
from pipeline_common import dataset_main

if __name__ == '__main__':
    raise SystemExit(dataset_main('noisy'))
