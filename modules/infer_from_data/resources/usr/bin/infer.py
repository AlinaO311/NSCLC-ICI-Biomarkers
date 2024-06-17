#!/usr/bin/env python3

import os
import argparse
import glob
from pathlib import Path
from typing import Optional

from trainer import Trainer

cwd=os.getcwd().split('work', 1)[0]


def main(experiment_folder: Path, data_path: Path, output_file: Path) -> None:
    assert output_file.suffix == ".csv", "ERROR: Output file does not have `.csv` extension! Exiting."
    model_config_path = experiment_folder
    print('model_config_path ', model_config_path)
    pathlist = os.path.dirname(experiment_folder).split('/')[:-1]
    print('pathlist', pathlist)
    pathlist = ['/' if x == '' else x for x in pathlist]
    model_path = os.path.join(cwd, *pathlist, 'model')
    print('model_path', model_path)
    trainer = Trainer(model_config_path, resume_model=model_path, data_path=data_path)
    trainer.predict(output_file)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Script to load a previously trained model and perform"
        "prediction on given data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_folder",
        type=Path,
        help="Path to the experiment folder of the trained model." # output/models/xgboost_model/
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data (.csv file) on which to run prediction.", # test_data.csv
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to file where to save the output data with predictions. "
        "Must end with `.csv`.",
        default="${params.infer_outfile}"
    )
    args = vars(parser.parse_args())

    main(**args)
