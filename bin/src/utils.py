#!/usr/bin/env python3

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from ruamel.yaml import YAML

import git

set_datetime = os.getenv('DATE_VALUE')

def read_config(config_path: Path) -> dict:
    """Extract a configuration dict from a .yaml file.

    Arguments:
        config_path -- The path to the .yaml config file.

    Returns:
        A dict containing the configuration.
    """
    with open(config_path) as file:
        yaml=YAML(typ='safe', pure=True)
        config = yaml.load(file)
    return config

def check_git_status()-> None:
    # Check status of git repo.
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        print(
            "\nWARNING!! The git repository is dirty. Commit all your changes before running scripts!\n"
        )

def save_git_status(path: Path) -> None:
    """Saves the status (git hash) of the git repository.

    Arguments:
        path -- The path to the folder where the git info will be saved.
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # Save git status info.
    with open(Path(path) / "git_info.txt", "w") as file:
        file.write(f"Git hash: {sha}")


def copy_config(config: dict, path_to_folder: Path, name) -> None:
    """Saves a copy of the configuration.

    Arguments:
        config -- The configuration dict to store.
        path_to_folder -- The path to folder of the intended location.
        name -- The intended name of the configuration file.
    """
    assert (
        os.path.isdir(path_to_folder)
    ), f"Failed to save config file. {path_to_folder} is not a folder."

    # Save the configuration to disk.
    with  open( os.path.join(path_to_folder ,f"{name}.yml"), "w") as f:
        yaml=YAML(typ='safe')
        yaml.dump(config, f)


def get_current_datetime() -> str:
    """Returns the current datetime as a string."""
    current_datetime = set_datetime.strip()
    return current_datetime


def create_directory(dir_path: Path) -> None:
    """Creates a directory at the given path."""
    Path(dir_path).mkdir(exist_ok=True, parents=True)


def prepare_save_folder(
    output_folder: Path,
    name: str,
    subfolders: List[str],
    configs: Dict[str, dict],
) -> Path:
    """Prepares a save folder.

    Arguments:
        output_folder -- The location of the folder.
        name -- The name the folder.
        subfolders -- Names of any subfolders.
        configs -- The configuration files to store.

    Returns:
        The path to the prepared folder.
    """
    print("Creating output folders.")
    output_dir = os.path.join(output_folder , name+'_'+get_current_datetime())

    # Create subfolders.
    for subfolder in subfolders:
        create_directory(os.path.join( output_dir , subfolder))

    # Store configs.
    for config_name in configs.keys():
        print('config_name', config_name)
        print('val' , configs[config_name])
        copy_config(configs[config_name], os.path.join(output_dir, "config"), config_name)

    # Store git status.
   # save_git_status(output_dir)

    return output_dir

