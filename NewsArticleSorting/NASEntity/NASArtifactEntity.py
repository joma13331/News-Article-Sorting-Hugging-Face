from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                    ["ingested_dir", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated", "message", "train_file_path",
                                    "prediction_file_path"])

DataPreprocessingArtifact = namedtuple("DataPreprocessingArtifact",
                                        ["is_preprocessed", "message", "train_dir_path",
                                        "pred_dir_path"])