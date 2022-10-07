import os
from datetime import datetime

def get_current_time_stamp() -> str:
    """
    FunctionName: get_current_time_stamp
    Description: This function returns the current date and time of the system

    returns: str
    """

    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"


# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "DataIngestion"
DATA_INGESTION_TRAINING_INPUT_DIR_KEY = "training_input_dir"
DATA_INGESTION_PREDICTION_INPUT_DIR_KEY = "prediction_input_dir"
DATA_INGESTION_TRAINING_DIR_NAME_KEY = "training_dir"
DATA_INGESTION_PREDICTION_DIR_NAME_KEY = "prediction_dir"

# Data Validation related variable
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR = "DataValidation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_VALIDATED_DIR_KEY = "validated_dir"
DATA_VALIDATION_TRAIN_DIR = "validated_train"
DATA_VALIDATION_PREDICTION_DIR = "validated_prediction"

# Cassandra Database related variables
CASSANDRA_DATABASE_CONFIG_KEY = "cassandra_database_config"
CASSANDRA_DATABASE_TRAINING_TABLE_NAME_KEY = "training_table_name"
CASSANDRA_DATABASE_PREDICTION_TABLE_NAME_KEY = "prediction_table_name"
CASSANDRA_DATABASE_SECURE_CONNECT_BUNDLE_DIRECTORY_KEY = "secure_connect_bundle_directory"
CASSANDRA_DATABASE_SECURE_CONNECT_BUNDLE_FILENAME_KEY = "secure_connect_bundle_filename"
CASSANDRA_DATABASE_ENVIRONMENT_KEY_CLIENT_ID_KEY = "environment_key_client_id"
CASSANDRA_DATABASE_ENVIRONMENT_KEY_CLIENT_SECRET_KEY = "environment_key_client_secret"
CASSANDRA_DATABASE_KEYSPACE_NAME = "keyspace_name"


# Data Preprocessing related variable
DATA_PREPROCESSING_CONFIG_KEY = "data_preprocessing_config"
DATA_PREPROCESSING_ARTIFACT_DIR = "DataPreprocessed"
DATA_PREPROCESSING_TRAIN_DIR = "preprocessed_train"
DATA_PREPROCESSING_PRED_DIR = "preprocessed_pred"
DATA_PREPROCESSING_OHE_MODEL_DIR_KEY = "ohe_model_dir"
DATA_PREPROCESSING_ONE_HOT_ENCODER_FILE_NAME_KEY = "one_hot_encoder_file_name"

# Model Trainer related variable

MODEL_TRAINER_MODEL_DETAILS_KEY = "model_detail"
MODEL_TRAINER_MODEL_NAME_KEY = "model_name"
MODEL_TRAINER_MODEL_TYPE_KEY = "model_type"
MODEL_TRAINER_MODEL_ARGS_KEY = "model_args"

MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_ARTIFACT_DIR = "ModelTraining"
MODEL_TRAINING_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"
MODEL_TRAINER_LEARNING_RATE_KEY = "learning_rate"

# Model evaluation related variables

MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_ARTIFACT_DIR = "ModelEvaluation"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_EVALUATION_MODEL_ACCESS_NAME = "model_access_name"

# Model pusher related variables

MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_ARTIFACT_DIR = "ModelPusher"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

# predictor related variables

PREDICTOR_ARTIFACT_DIR = "Prediction"
