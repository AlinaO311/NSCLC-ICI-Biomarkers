#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import Optional

from visualizer import Visualizer

cwd=os.getcwd().split('work', 1)[0]

def main(dir: Path, mutation_data: Path,  data_path: Optional[Path] = None) -> None:
    output_path = os.path.join(cwd, dir ,"DataPrep")
    visualizer = Visualizer(
        output_path=output_path,
        mutfile=mutation_data, 
        data_path=data_path,
    )
    visualizer.visualize()


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Script to visualize statustucs about data. Results are saved "
        "under <experiment_directory>/DataPrep/visualization/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data on which to run the analysis.",
    )
    parser.add_argument('--mutation_data', help='File of mutations of interest', required=False, default='mutations.txt')
    parser.add_argument("--dir", type=Path, help="Path to the output folder." )

    args = vars(parser.parse_args())

    main(**args)
