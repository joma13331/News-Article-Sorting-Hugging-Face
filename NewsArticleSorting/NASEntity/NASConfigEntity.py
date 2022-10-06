from collections import namedtuple


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
                                    ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionCongfig",
                                ["raw_input_dir", "ingested_dir"] 
                                )