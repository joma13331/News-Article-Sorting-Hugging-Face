import os
import sys
import yaml
import pickle
from NewsArticleSorting.NASException import NASException

def read_yaml_file(file_path:str)->dict:
    """
    FunctionName: read_yaml_file
    Parameter: file_path: str
    Description: Reads a YAML file and returns the contents as a dictionary.
    
    returns: dict
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NASException(e,sys) from e

def save_model(model, model_file_path):
    """
    :Function Name: save_model
    :Description: This method saves the passed model to the given path

    :param model: The model to save.
    :param model_file_path: path where 
    :return: None
    """
    try:
        # Saving the model to the path 
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)

    except Exception as e:
        raise NASException(e, sys) from e

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise NASException(e,sys)

def string_to_tuple(text):
    try:
        
        splitted_text = text[1:-1].split(",")
        splitted_int = tuple([int(float(num))for num in splitted_text])

        return splitted_int
    except Exception as e:
        raise NASException(e, sys) from e