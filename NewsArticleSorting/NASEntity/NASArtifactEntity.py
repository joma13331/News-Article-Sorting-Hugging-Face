from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                    ["ingested_dir", "is_ingested", "message"])