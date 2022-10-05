import os
import logging
import pandas as pd

from NewsArticleSorting.NASConstants import get_current_time_stamp


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

def get_log_dataframe(file_path) -> pd.DataFrame:
    """
    FunctionName: get_log_dataframe
    Description: Converts the logfiles into pandas Datframe with columns:
                 Time stamp, Log Level, line number, file name, function name, message
    
    returns: Pandas DatFrame with only one column log_message
    """

    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Log Level","line number","file name","function name","message"]
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["message"]

    return log_df[["log_message"]]