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

    def get_data_ingestion_config(self)-> DataIngestionConfig:

        try:
            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            if self.is_training:

                raw_input_dir = os.path.join(ROOT_DIR,
                                            data_ingestion_config[DATA_INGESTION_TRAINING_INPUT_DIR_KEY])
                
                ingested_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            DATA_INGESTION_ARTIFACT_DIR,
                                            self.time_stamp,
                                            data_ingestion_config[DATA_INGESTION_TRAINING_DIR_NAME_KEY])

            else:
                raw_input_dir = os.path.join(ROOT_DIR, data_ingestion_config[DATA_INGESTION_PREDICTION_INPUT_DIR_KEY])
                ingested_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            DATA_INGESTION_ARTIFACT_DIR,
                                            self.time_stamp,
                                            data_ingestion_config[DATA_INGESTION_PREDICTION_DIR_NAME_KEY])


            data_ingestion_config = DataIngestionConfig(
                raw_input_dir=raw_input_dir,
                ingested_dir=ingested_dir
            )

            return data_ingestion_config

        except Exception as e:
            raise NASException(e, sys) from e 

    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            schema_file_path = os.path.join(ROOT_DIR, data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                            data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])

            validated_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            DATA_VALIDATION_ARTIFACT_DIR,
                                            self.time_stamp,
                                            data_validation_config[DATA_VALIDATION_VALIDATED_DIR_KEY])
            if self.is_training:
                validated_train_dir = os.path.join(validated_dir, DATA_VALIDATION_TRAIN_DIR)
                validated_prediction_dir = None
            else:
                validated_train_dir = None
                validated_prediction_dir = os.path.join(validated_dir, DATA_VALIDATION_PREDICTION_DIR)
            

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                validated_dir=validated_dir,
                validated_train_dir=validated_train_dir,
                validated_prediction_dir=validated_prediction_dir
            )

            logging.info(f"The Data Validation Config: {data_validation_config}")

            return data_validation_config
            
        except Exception as e:
            raise NASException(e, sys) from e

    def get_cassandra_database_config(self)-> CassandraDatabaseConfig:

        try:
            cassandra_db_config = self.config_info[CASSANDRA_DATABASE_CONFIG_KEY]

            file_path_secure_connect = os.path.join(ROOT_DIR,
                                        self.config_info[TRAINING_PIPELINE_CONFIG_KEY][TRAINING_PIPELINE_NAME_KEY],
                                        cassandra_db_config[CASSANDRA_DATABASE_SECURE_CONNECT_BUNDLE_DIRECTORY_KEY],
                                        cassandra_db_config[CASSANDRA_DATABASE_SECURE_CONNECT_BUNDLE_FILENAME_KEY])
            if self.is_training:
                table_name = cassandra_db_config[CASSANDRA_DATABASE_TRAINING_TABLE_NAME_KEY]
            
            else:
                table_name = cassandra_db_config[CASSANDRA_DATABASE_PREDICTION_TABLE_NAME_KEY]

            cassandra_client_id =os.getenv(cassandra_db_config[CASSANDRA_DATABASE_ENVIRONMENT_KEY_CLIENT_ID_KEY])
            cassandra_client_secret = os.getenv(cassandra_db_config[CASSANDRA_DATABASE_ENVIRONMENT_KEY_CLIENT_SECRET_KEY])

            keyspace_name = cassandra_db_config[CASSANDRA_DATABASE_KEYSPACE_NAME]
            

            return CassandraDatabaseConfig(
                file_path_secure_connect=file_path_secure_connect,
                table_name=table_name,
                cassandra_client_id=cassandra_client_id,
                cassandra_client_secret=cassandra_client_secret,
                keyspace_name=keyspace_name
            )
        except Exception as e:
            raise NASException(e, sys) from e