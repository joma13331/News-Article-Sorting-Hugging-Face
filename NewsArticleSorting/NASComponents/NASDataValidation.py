import os
import re
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
from NewsArticleSorting.NASDatabase.NASCassandraDB import NASCassandraDB

from NewsArticleSorting.NASUtils.utils import read_yaml_file
from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASEntity.NASConfigEntity import DataValidationConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataValidationArtifact, DataIngestionArtifact


class NASDataValidation:
    """
    Class Name: NASDataValidation
    Description: Includes all the methods that are needed to check whether the data has been provide in the correct format.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self, 
    data_validation_config: DataValidationConfig,
    data_ingestion_artifact: DataIngestionArtifact,
    cassandra_db_operator: NASCassandraDB=None,
    use_database: bool=False) -> None:

        try:

            logging.info( f"{'*'*20} Data Validation log started {'*'*20}")

            self.use_database = use_database
            self.db_operator = cassandra_db_operator
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_dict = read_yaml_file(self.data_validation_config.schema_file_path)

            if self.data_validation_config.validated_prediction_dir is None:
                os.makedirs(self.data_validation_config.validated_train_dir, exist_ok=True)

            else: 
                os.makedirs(self.data_validation_config.validated_prediction_dir,exist_ok=True)
            
        except Exception as e:
            raise NASException(e, sys) from e

    def regex_file_name(self) -> re.Pattern:
        """
        Method Name: regex_file_name
        Description: This method return the regex pattern of filenames that are acceptable.

        return: re.Pattern: Acceptable Regex pattern
        """
        try:
            # From the sample filename provided ontaining the regex for correct file name
            file_name_initial = self.schema_dict['SampleFileName'].split('_')[0]
            regex = re.compile(r'{}_[0123]\d[01]\d[12]\d\d\d_[012]\d[0-5]\d[0-5]\d.csv'.format(file_name_initial))
            return regex

        except Exception as e:
            raise NASException(e, sys) from e

    
    def check_valid_filename(self, regex: re.Pattern) -> Tuple[bool, str, List[str]]:
        """
        Method Name: check_valid_filename
        parameter: regex- the regex pattern to match
        Description: This method checks and removes the filenames from being considered for futher processing
                    violate aproved file naming convention.

        return: tuple of 1. bool: whether any filename with acceptable filenames are present
                         2. str: the relevant message
                         3. list: list of validated filenames
        """
        try:
            ingested_dir = self.data_ingestion_artifact.ingested_dir
            raw_files = os.listdir(ingested_dir)
            valid_filenames = []
            # Adding all the files with correct filename to the apprpriate list
            for filename in raw_files:
                if re.match(regex, filename):
                    valid_filenames.append(filename)
            
            # Checking to see the there are any valid filenames
            if valid_filenames:
                cont = True
                message = f"There are files with valid filenames which were ingested"
                logging.info(message)
                
            else:
                cont = False
                message = "The are no files with valid filenames which were ingested"
                logging.info(message)
                
            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def check_valid_columns(self, filenames: List[str]) -> Tuple[bool, str, List[str]]:
        """
        Method Name: check_valid_columns
        parameter: filenames-list of filenames to validate
        Description: This method checks and removes the filenames from being considered for futher processing
                    violate valid columns.

        return: tuple of 1. bool: whether any filename with acceptable/all columns are present
                         2. str: the relevant message
                         3. list: list of validated filenames
        """
        try:
            
            valid_filenames = []
            schema_columns = []

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn']      
            schema_columns.extend([col_name for col_name in schema_input_columns])

            # Adding the target column only if we are performing training
            if self.data_validation_config.validated_prediction_dir is None:
                schema_columns.extend([col_name for col_name in schema_target_column])

            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)

                df = pd.read_csv(file_path)
                df_columns = list(df.columns.values)

                # sorting as it will help us check that whether all the columns in the file provided and in the schema are same         
                df_columns.sort()
                schema_columns.sort()

                # checking to see if the columns are same for both the csv file and in the schema
                if df_columns == schema_columns:
                    valid_filenames.append(filename)

            # Checking to see the there are any valid filenames
            if valid_filenames:
                cont = True
                message = "There are files with valid columns in the files that were ingested"
                logging.info(message)
            else:
                cont = False
                message = "There are no files with valid columns in the files that were ingested"
                logging.info(message)

            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def check_correct_datatypes(self, filenames: List[str]) -> Tuple[bool, str, List[str]]:
        """
        Method Name: check_correct_datatypes
        parameter: filenames-list of filenames to validate

        Description: This method checks and removes the filenames from being considered for futher processing
                    violate valid datatypes of all columns.

        return: tuple of 1. bool: whether any filename with valid datatypes of all columns are present
                         2. str: the relevant message
                         3. list: list of validated filenames
        """
        
        try:
            
            valid_filenames = []

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn'] 

            schema_columns = schema_input_columns

            # Adding the Key-Value pair for target if we are performing training
            if self.data_validation_config.validated_prediction_dir is None:
                schema_columns.update(schema_target_column)


            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                
                df = pd.read_csv(file_path)
                # Obtaining a dictionary of the column names and their datatypes
                df_columns = df.dtypes.to_dict()

                # Changing the datatypes in the dictionary so that it maybe compared to the one from schema
                for key, val in df_columns.items():
                    if val==np.dtype('object_'):
                        df_columns[key] = 'text'
                    if val==np.dtype('int64'):
                        df_columns[key] = 'int'
                
                # Checking whether the datatypes are as they should be.
                if schema_columns==df_columns:
                    valid_filenames.append(filename)

            # Checking to see the there are any valid filenames
            if valid_filenames:
                cont = True
                message = "There are files with valid datatypes in the files that were ingested"
                logging.info(message)
            else:
                cont = False
                message = "There are no files with invalid columns in the files that were ingested"
                logging.info(message)

            return (cont, message, valid_filenames)                    

        except Exception as e:
            raise NASException(e, sys) from e

    def check_correct_domain_values(self,filenames: List[str]) -> Tuple[bool, str, List[str]]:
        """
        Method Name: check_correct_domain_values
        parameter: filenames-list of filenames to validate

        Description: This method checks and removes the filenames from being considered for futher processing
                    violate valid domain values in the target column.

        return: tuple of 1. bool: whether any filename with valid domain values in the target column are present
                         2. str: the relevant message
                         3. list: list of validated filenames
        """
        try:
            valid_filenames = []

            # Obtaining the allowed domain values of the target from theschema
            schema_target_vals = self.schema_dict["TargetColumnValues"]
            schema_target_vals.sort()

            for filename in filenames:

                file_path =os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                df = pd.read_csv(file_path)
                
                # Obtaining the domain values of the target in the files provided for training
                df_target_vals = list(df['Category'].unique())

                df_target_vals.sort()

                # Checking to see if the domain values in the files are same as expected
                if schema_target_vals==df_target_vals:
                    valid_filenames.append(filename)

            # Checking to see the there are any valid filenames
            if valid_filenames:
                cont = True
                message = "The target columns only has value that were specified in the schema"
            else:
                cont = False
                message = "The target columns has other values than were specified in the schema"
            
            logging.info(message)

            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def combine(self, filenames: List[str]) -> Tuple[bool, str, str, str]:
        """
        Method Name: combine
        parameter: filenames-list of filenames to validate

        Description: This method all the validated files into a single csv file..

        return: tuple of 1. bool: whether combining all the files is successful
                         2. str: the relevant message
                         3. str: validated train file path
                         4. str: validated prediction file path
        """
        try:
            list_dataframes = []

            # Combining all the data from the files to be used for training into a single dataframe
            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                df = pd.read_csv(file_path)
                list_dataframes.append(df)

            final_dataframe = pd.concat(list_dataframes, ignore_index=True)

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn'] 

            schema_columns = schema_input_columns

            # Actions when we are training
            if self.data_validation_config.validated_prediction_dir is None:
                
                validated_train_file_path = os.path.join(self.data_validation_config.validated_train_dir, "train_dataset.csv")
                validated_prediction_file_path = None
                
                # If database is to be used then perform the appropriate datbase operation
                if self.use_database:
                    schema_columns.update(schema_target_column)
                    self.db_operator.create_table(column_names=schema_columns)             
                    self.db_operator.insert_valid_data(df=final_dataframe)
                    self.db_operator.data_db_to_csv(validated_file_path=validated_train_file_path)
                    self.db_operator.terminate_session()
                else:
                    final_dataframe.to_csv(validated_train_file_path, index=False)            

                is_successful = True
                message = f"All the relevant data has been stored in path {validated_train_file_path} for"\
                f"training"
                logging.info(message)
            # Actions when we are predicting
            else:
                
                validated_train_file_path = None
                validated_prediction_file_path = os.path.join(self.data_validation_config.validated_prediction_dir, "prediction_dataset.csv")

                # If database is to be used then perform the appropriate datbase operation
                if self.use_database:
                    schema_columns.update(schema_target_column)
                    self.db_operator.create_table(column_names=schema_columns)             
                    self.db_operator.insert_valid_data(df=final_dataframe)
                    self.db_operator.data_db_to_csv(validated_file_path=validated_prediction_file_path)
                    self.db_operator.terminate_session()
                else:
                    final_dataframe.to_csv(validated_prediction_file_path, index=False)

                is_successful = True
                message = f"All the relevant data has been stored in path {validated_prediction_file_path} for"\
                f"prediction"
                logging.info(message)

            return (is_successful, message, validated_train_file_path, validated_prediction_file_path)

        except Exception as e:
            raise NASException(e, sys) from e


    def initiate_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_validation
        Description: Combines all the relevant methods to achieve data validation.

        return: Data Validation Artifact: contains all the relevant information for futher processing of the data
        """
        try:
            
            regex = self.regex_file_name()
            cont, message, filenames = self.check_valid_filename(regex=regex)

            # Progessing further only if valid filenames are provided
            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    test_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact

            cont, message, filenames = self.check_valid_columns(filenames=filenames)

            # Progessing further only if valid files are provided
            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact
            
            cont, message, filenames = self.check_correct_datatypes(filenames=filenames)

            # Progessing further only if valid datatypes are provided
            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact

            # Check the domain values of target if we are training
            if self.data_validation_config.validated_prediction_dir is None:
                cont, message, filenames = self.check_correct_domain_values(filenames=filenames)

                # Checking to see if valid domain values are present 
                if not cont:
                    data_validation_artifact = DataValidationArtifact(
                        is_validated=cont,
                        message=message,
                        train_file_path=None,
                        prediction_file_path=None
                    )
                    return data_validation_artifact
            
            # Combining all the relevant files into a single file and saving at correct location
            is_validated, message, train_file_path,  prediction_file_path = self.combine(
                filenames=filenames)

            data_validation_artifact = DataValidationArtifact(
                is_validated=is_validated,
                message=message,
                train_file_path=train_file_path,
                prediction_file_path=prediction_file_path
            )

            logging.info(f"The Data Validation Artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise NASException(e, sys) from e
