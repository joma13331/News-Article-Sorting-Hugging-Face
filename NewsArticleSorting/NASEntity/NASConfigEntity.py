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