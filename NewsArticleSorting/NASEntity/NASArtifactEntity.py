from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                    ["ingested_dir", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated", "message", "train_file_path",
                                    "prediction_file_path"])

DataPreprocessingArtifact = namedtuple("DataPreprocessingArtifact",
                                        ["is_preprocessed", "message", "train_dir_path",
                                        "pred_dir_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",
                                ["is_trained", "message", "trained_model_path", "model_name",
                                "optimizer", "lr"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact",
                                ["is_accepted", "message",  "model_evaluation_result_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact",
                                ["is_pushed", "message"])

PredictorArtifact = namedtuple("PredictorArtifact",
                                    ["is_prediction_done", "message"])