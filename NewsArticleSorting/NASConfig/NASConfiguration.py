import os
import sys

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASEntity.NASConfigEntity import *

from NewsArticleSorting.NASUtils.utils import read_yaml_file
from NewsArticleSorting.NASConstants import *


class NASConfiguration:

    def __init__(self,
    config_file_path: str=CONFIG_FILE_PATH,
    current_time_stamp: str=CURRENT_TIME_STAMP,
    is_training=True) -> None:

        try:
            self.config_info = read_yaml_file(config_file_path)
            self.time_stamp = current_time_stamp
            self.is_training = is_training

            self.training_pipeline_config = self.get_training_pipeline_config()


        except Exception as e:
            raise NASException(e, sys) from e

    
    def get_training_pipeline_config(self)-> TrainingPipelineConfig:

        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
            training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
            training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])

            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            return training_pipeline_config

        except Exception as e:
            raise NASException(e, sys) from e
