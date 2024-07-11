from dataset_utils import *
import os

def get_json_files(folder_path):
    """
    This function retrieves all JSON files from the specified folder and returns their filenames.

    Parameters:
    folder_path (str): The path to the folder containing JSON files.

    Returns:
    list: A list of filenames.
    """
    all_files = os.listdir(folder_path)
    json_files = [file for file in all_files if file.endswith('.json')]
    return json_files

