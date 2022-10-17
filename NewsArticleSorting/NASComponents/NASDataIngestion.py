import os
import sys
import shutil
from typing import Tuple

from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASEntity.NASConfigEntity import DataIngestionConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataIngestionArtifact
from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

class NASDataIngestion:
    """
    ClassName: NASDataIngestion
    Description: Includes all the methods that are needed to bring the data from the 
                provided source to the desired format in the required location

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            logging.info( f"{'*'*20} Data Ingestion log started {'*'*20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise NASException(e, sys) from e

    def copy_raw_files_artifact_directory(self, expected_extension: str) -> Tuple[bool, str]:
        """
        MethodName: copy_raw_files_artifact_directory
        Parameter: 
            expected_extension: The accepted extension of the input files
        Description: This method copies all files of the specified extension to the desired folder

        returns: Boolean: Whether the data ingestion was successful
                 str: message - the message detailing the transfer to ingestion dir
        """
        try:

            list_raw_files = os.listdir(self.data_ingestion_config.raw_input_dir)

            # Checking to see if any filesare provided in the raw file directory
            if len(list_raw_files) == 0:
                is_successful = False
                message = f"The Ingestion process failed as there are no files provided in {self.data_ingestion_config.raw_input_dir}"
                return (is_successful, message) 

            # Copying all the available file with the expected extension to the data ingestion folder
            dest = self.data_ingestion_config.ingested_dir            
            for filename in list_raw_files:
                if filename.endswith(expected_extension):
                    src = os.path.join(self.data_ingestion_config.raw_input_dir, filename)
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy2(src=src, dst=dest)
            
            is_successful = True
            message = f"All the files in {self.data_ingestion_config.raw_input_dir} with"\
            f"extension {expected_extension} have been transferred to {self.data_ingestion_config.ingested_dir}"
            
            logging.info(message)
            
            return (True, message)

        except Exception as e:
            raise NASException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        MethodName: initiate_data_ingestion
        Description: This method compiles and executes all necessary Ingestion tasks that
                    are needed

        returns: data_ingestion_artifact- Containing relevant information about the data ingestion process.
        """
        try:
            
            # Calling the method to copy all relevant file in the raw folder to data ingestion folder
            is_successful, message = self.copy_raw_files_artifact_directory(expected_extension= ".csv")
            
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_dir=self.data_ingestion_config.ingested_dir,
                is_ingested=is_successful,
                message=message
            )

            logging.info(f"the Data Ingestion Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise NASException(e, sys) from e


