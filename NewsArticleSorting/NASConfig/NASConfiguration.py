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


    def get_data_preprocessing_config(self)-> DataPreprocessingConfig:
        try:
            data_preprocessing_config = self.config_info[DATA_PREPROCESSING_CONFIG_KEY]
            
            if self.is_training:
                preprocessed_train_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            DATA_PREPROCESSING_ARTIFACT_DIR,
                                            self.time_stamp,                                        
                                            DATA_PREPROCESSING_TRAIN_DIR)

                preprocessed_pred_dir =None

            else:
                preprocessed_train_dir = None
                preprocessed_pred_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            DATA_PREPROCESSING_ARTIFACT_DIR,
                                            self.time_stamp,                                        
                                            DATA_PREPROCESSING_PRED_DIR)
            
            ohe_file_path = os.path.join(self.training_pipeline_config.artifact_dir,
                                        DATA_PREPROCESSING_ARTIFACT_DIR,
                                        data_preprocessing_config[DATA_PREPROCESSING_OHE_MODEL_DIR_KEY],
                                        data_preprocessing_config[DATA_PREPROCESSING_ONE_HOT_ENCODER_FILE_NAME_KEY])

            os.makedirs(os.path.join(self.training_pipeline_config.artifact_dir,
                                        DATA_PREPROCESSING_ARTIFACT_DIR,
                                        data_preprocessing_config[DATA_PREPROCESSING_OHE_MODEL_DIR_KEY]), exist_ok=True)            
            
            return DataPreprocessingConfig(
                preprocessed_train_dir=preprocessed_train_dir,
                preprocessed_pred_dir=preprocessed_pred_dir,
                ohe_file_path=ohe_file_path,
            )

        except Exception as e:
            raise NASException(e, sys) from e

    def get_model_training_config(self)-> ModelTrainingConfig:
        try:
            model_training_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            model_config_file_path = os.path.join(ROOT_DIR,
                                    model_training_config[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                                    model_training_config[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY])
            
            model_info = read_yaml_file(model_config_file_path)

            trained_model_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            MODEL_TRAINER_ARTIFACT_DIR,
                                            self.time_stamp,
                                            model_training_config[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
                                            )
                                            
            os.makedirs(trained_model_dir, exist_ok=True)

            trained_model_path = os.path.join(trained_model_dir, "model.bin")

            base_accuracy = model_training_config[MODEL_TRAINER_BASE_ACCURACY_KEY]

            max_seq_length = model_training_config[MODEL_TRAINER_MAX_SEQ_LENGTH_KEY]

            padding_type = model_training_config[MODEL_TRAINER_PADDING_TYPE_KEY]

            truncation = model_training_config[MODEL_TRAINER_TRUNCATION_KEY]

            hyperparameter_tuning_info = model_info[MODEL_TRAINER_HYPERPARAMETER_TUNING_KEY]

            models = hyperparameter_tuning_info[MODEL_TRAINER_MODEL_NAMES_KEY]

            optimizers = hyperparameter_tuning_info[MODEL_TRAINER_OPTIMIZERS_KEY]

            learning_rate_start = hyperparameter_tuning_info[MODEL_TRAINER_LEARNING_RATE_KEY][MODEL_TRAINER_LEARNING_START_KEY]

            learning_rate_end = hyperparameter_tuning_info[MODEL_TRAINER_LEARNING_RATE_KEY][MODEL_TRAINER_LEARNING_END_KEY]

            hyperparameter_tuning_epochs = hyperparameter_tuning_info[MODEL_TRAINER_HYPERPARAMETER_TUNING_EPOCHS_KEY]

            num_train_epochs = model_info[MODEL_TRAINER_MODEL_ARGS_KEY][MODEL_TRAINING_NUM_TRAIN_EPOCHS_KEY]

            train_batch_size = model_info[MODEL_TRAINER_MODEL_ARGS_KEY][MODEL_TRAINING_TRAIN_BATCH_SIZE_KEY]

            input_feature = model_training_config[MODEL_TRAINER_INPUT_FEATURE_KEY]

            no_of_models_to_check = hyperparameter_tuning_info[MODEL_TRAINING_NUMBER_OF_MODELS_TO_CHECK]
            
            return ModelTrainingConfig(
                trained_model_path=trained_model_path,
                base_accuracy=base_accuracy,
                max_seq_length=max_seq_length,
                padding_type=padding_type,
                truncation=truncation,
                models=models,
                optimizers=optimizers,
                learning_rate_start=learning_rate_start,
                learning_rate_end=learning_rate_end,
                hyperparameter_tuning_epochs=hyperparameter_tuning_epochs,
                num_train_epochs=num_train_epochs,
                train_batch_size=train_batch_size,
                input_feature=input_feature,
                no_of_models_to_check=no_of_models_to_check
            )

        except Exception as e:
            raise NASException(e, sys) from e

    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]

            model_evaluation_result_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                            MODEL_EVALUATION_ARTIFACT_DIR,
                                            self.time_stamp,
                                            model_evaluation_config[MODEL_EVALUATION_RESULT_DIR_KEY])

            os.makedirs(model_evaluation_result_dir, exist_ok=True)
            
            model_evaluation_result_path = os.path.join(model_evaluation_result_dir,"model_evaluation_result.yaml")

            base_accuracy = model_evaluation_config[MODEL_EVALUATION_BASE_ACCURACY_KEY]
            
            eval_batch_size = model_evaluation_config[MODEL_EVALUATION_BATCH_SIZE_KEY]
            
            return ModelEvaluationConfig(
                model_evaluation_result_path=model_evaluation_result_path,
                base_accuracy=base_accuracy,
                eval_batch_size=eval_batch_size
            )

        except Exception as e:
            raise NASException(e,sys) from e

    def get_model_pusher_config(self)-> ModelPusherConfig:
        try:
            model_pusher_config = self.config_info[MODEL_PUSHER_CONFIG_KEY]

            model_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                    model_pusher_config[MODEL_PUSHER_MODEL_EXPOSED_DIR_KEY],
                                    )
            
            os.makedirs(model_dir, exist_ok=True)

            deployed_model_path = os.path.join(model_dir, model_pusher_config[MODEL_PUSHER_DEPLOYED_MODEL_FILENAME_KEY])

            deployed_model_details_path = os.path.join(
                model_dir,
                model_pusher_config[MODEL_PUSHER_DEPLOYED_MODEL_DETAILS_KEY]
            )

            return ModelPusherConfig(
                deployed_model_path=deployed_model_path,
                deployed_model_details_path=deployed_model_details_path
            )


        except Exception as e:
            raise NASException(e, sys) from e
    