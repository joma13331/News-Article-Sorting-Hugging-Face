from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                    ["ingested_dir", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated", "message", "train_file_path",
                                    "prediction_file_path"])