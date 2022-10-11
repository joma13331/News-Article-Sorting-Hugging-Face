from collections import namedtuple


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
                                    ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionCongfig",
                                ["raw_input_dir", "ingested_dir"] 
                                )

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path", "validated_dir",
 "validated_train_dir", "validated_prediction_dir"]) 

CassandraDatabaseConfig = namedtuple("CassandraDatabaseConfig",
                        ["file_path_secure_connect","table_name",
                        "cassandra_client_id", "cassandra_client_secret",
                        "keyspace_name"])    

DataPreprocessingConfig = namedtuple("DataPreprocessingConfig", ["preprocessed_train_dir",
 "preprocessed_pred_dir", "ohe_file_path"])   

ModelTrainingConfig = namedtuple("ModelTrainingConfig",
                                ["trained_model_path", "base_accuracy", "max_seq_length",
                                "padding_type", "truncation", "models", "optimizers", 
                                "learning_rate_start", "learning_rate_end", 
                                "hyperparameter_tuning_epochs", "num_train_epochs",
                                "train_batch_size",
                                "input_feature","no_of_models_to_check"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig",
                                ["model_evaluation_result_path", "base_accuracy", "eval_batch_size"])

ModelPusherConfig = namedtuple("ModelPusherConfig",
                            ["deployed_model_path", "deployed_model_details_path"])