import os
import shutil
from data_api.db import get_all_models


def empty_data_dirs(model_dir, scaler_dir):
    """
    This function deletes all files in the specified directories
    """
    # Empty the existing directories
    shutil.rmtree(model_dir)
    shutil.rmtree(scaler_dir)
    # Create new directories
    os.makedirs(model_dir)
    os.makedirs(scaler_dir)
    return True


def delete_model_files_not_in_db(engine, model_dir, scaler_dir):
    """
    Deletes all the model (.h5) and scaler files which do not belong to a model in the database
    """
    model_names = []
    for model in get_all_models(engine):
        model_names.append(model[0].model_name)
    for file in os.listdir(model_dir):
        filename = os.fsdecode(file)
        if filename[:-3] not in model_names:
            os.remove(os.path.join(model_dir, filename))

    for file in os.listdir(scaler_dir):
        filename = os.fsdecode(file)
        if filename not in model_names:
            os.remove(os.path.join(scaler_dir, filename))


def delete_model_files(model_name, model_dir, scaler_dir):
    """
    This function deletes all files for the specified model name
    """
    model_file_name = model_name + ".h5"
    try:
        os.remove(os.path.join(model_dir, model_file_name))
    except FileNotFoundError as error:
        print(error)
    try:
        os.remove(os.path.join(scaler_dir, model_name))
    except FileNotFoundError as error:
        print(error)
