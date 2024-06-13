#!/usr/bin/env python3

import sys
import argparse
import os
import json
import glob
#import ruamel.yaml
#from ruamel.yaml.scalarstring import SingleQuotedScalarString, DoubleQuotedScalarString
#from datetime import datetime
from pathlib import Path
from preprocessor import Preprocessor
#from utils import read_config, get_current_datetime

cwd=os.getcwd().split('work', 1)[0]

def main(config_path: Path) -> None:
    #check_git_status()
    preorcessor = Preprocessor(config_path)
    preorcessor.process()


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Script to prepare data for training or predicton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        required=True,
        type=Path,
        default=Path(os.path.join(cwd, "${params.output_dir}", "configs/preprocess/preprocess_config.yml")),
        help="Path to preprocess config file.",
    )
    parser.add_argument('--remove_cols', help='names of cols to drop, ex: HISTOLOGY,PFS_MONTHS',required=False, default=None)
    parser.add_argument('--model_type', help='ML model, default: xgboost',required=False, default='xgboost')
    parser.add_argument('--outdir', help='name of output directory, default: outdir',required=False, default='outdir')

    args = parser.parse_args()

    main(sys.argv[2])




