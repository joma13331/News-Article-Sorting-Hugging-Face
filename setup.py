from setuptools import setup, find_packages
from typing import List

# Variable Declaration for setup function
PROJECT_NAME = "News-Article-Sorting"
VERSION = "0.0.1"
AUTHOR = "Jobin Mathew"
DESCRIPTION = "iNeuron Internship project based on Kaggle Dataset to Classify News Articles"

REQUIREMENT_FILE_NAME = "requirements.txt"

HYPHEN_E_DOT = "-e ."

def get_requirements_list() -> List[str]:
    """
    Function Name: get_requirements_list
    Description: This function returns list of requirements mentioned 
    in requirements.txt file

    returns List[str]
    """
    with open(REQUIREMENT_FILE_NAME, encoding="utf-16") as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
            print(requirement_list)
        return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list()
)

