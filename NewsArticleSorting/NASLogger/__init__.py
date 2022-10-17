import os
import logging
from typing import List
import pandas as pd

from NewsArticleSorting.NASConstants import get_current_time_stamp
from NewsArticleSorting.NASConstants import ROOT_DIR


def get_log_filename() -> str:
    """
    FunctionName: get_log_filename
    Description: This function will return the filename of the log file that 
                 will be created for a particular logging instant.

    returns: str
    """
    return f"log_{get_current_time_stamp()}.log"

LOGS_DIR = "NASLogs"
LOG_FILENAME = get_log_filename()

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILENAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format='[%(asctime)s^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s]',
    level=logging.INFO
    )

def get_log_list(num_logs:int = 50) -> List[str]:
    """
    FunctionName: get_log_dataframe
    Description: Converts the logfiles into pandas Datframe with columns:
                 Time stamp, Log Level, line number, file name, function name, message
    
    returns: Pandas DatFrame with only one column log_message
    """

    log_dir_path = os.path.join(ROOT_DIR, LOGS_DIR)
    filenames = sorted(os.listdir(log_dir_path), reverse=True)
    log_data = []

    count = 0
    for filename in filenames:
        filename_path = os.path.join(log_dir_path, filename)
        with open(filename_path) as log_file:
            logs_file = log_file.readlines()
            if not logs_file:
                continue
            if (count + len(logs_file)) >= num_logs:
                [log_data.append(f"{line.split('^;')[0]}: {line.split('^;')[1]}:: {line.split('^;')[-1]}") for line in logs_file[:(num_logs-count)]] 
                break
            else:
                [log_data.append(f"{line.split('^;')[0]}: {line.split('^;')[1]}:: {line.split('^;')[-1]}") for line in logs_file]
                count += len(logs_file)

    
    return log_data

if __name__ == "__main__":
    print(len(get_log_list()))