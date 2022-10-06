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
from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASEntity.NASConfigEntity import DataValidationConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataValidationArtifact, DataIngestionArtifact


class NASDataValidation:

    def __init__(self, 
    data_validation_config: DataValidationConfig,
    data_ingestion_artifact: DataIngestionArtifact,
    cassandra_db_operator: NASCassandraDB=None,
    use_database: bool=False) -> None:

        try:

            logging.info( f"{'*'*20} Data Ingestion log started {'*'*20}")

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
        try:
            file_name_initial = self.schema_dict['SampleFileName'].split('_')[0]
            regex = re.compile(r'{}_[0123]\d[01]\d[12]\d\d\d_[012]\d[0-5]\d[0-5]\d.csv'.format(file_name_initial))
            return regex

        except Exception as e:
            raise NASException(e, sys) from e

    
    def check_valid_filename(self, regex: re.Pattern) -> Tuple[bool, str, List[str]]:
        try:
            ingested_dir = self.data_ingestion_artifact.ingested_dir
            raw_files = os.listdir(ingested_dir)
            valid_filenames = []
            for filename in raw_files:
                if re.match(regex, filename):
                    valid_filenames.append(filename)
            
            if valid_filenames:
                cont = True
                message = f"There are files with valid filenames"
                
            else:
                cont = False
                message = "The are no files with valid filenames"
                
            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def check_valid_columns(self, filenames: List[str]) -> Tuple[bool, str, List[str]]:
        try:
            
            valid_filenames = []
            schema_columns = []

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn']      
            schema_columns.extend([col_name for col_name in schema_input_columns])
            if self.data_validation_config.validated_prediction_dir is None:
                schema_columns.extend([col_name for col_name in schema_target_column])

            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)

                df = pd.read_csv(file_path)
                df_columns = list(df.columns.values)
         
                df_columns.sort()
                schema_columns.sort()

                if df_columns == schema_columns:
                    valid_filenames.append(filename)

            if valid_filenames:
                cont = True
                message = "There are files with valid columns"
            else:
                cont = False
                message = "There are no files with valid columns"

            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def check_correct_datatypes(self, filenames: List[str]) -> Tuple[bool, str, List[str]]:
        try:
            
            valid_filenames = []

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn'] 

            schema_columns = schema_input_columns

            if self.data_validation_config.validated_prediction_dir is None:
                schema_columns.update(schema_target_column)


            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                
                df = pd.read_csv(file_path)
                df_columns = df.dtypes.to_dict()

                for key, val in df_columns.items():
                    if val==np.dtype('object_'):
                        df_columns[key] = 'text'
                    if val==np.dtype('int64'):
                        df_columns[key] = 'int'
                

                if schema_columns==df_columns:
                    valid_filenames.append(filename)

            if valid_filenames:
                cont = True
                message = "There are files with valid columns"
            else:
                cont = False
                message = "There are no files with valid columns"

            return (cont, message, valid_filenames)                    

        except Exception as e:
            raise NASException(e, sys) from e

    def check_correct_domain_values(self,filenames: List[str]) -> Tuple[bool, str, List[str]]:
        try:
            valid_filenames = []

            schema_target_vals = self.schema_dict["TargetColumnValues"]
            schema_target_vals.sort()

            for filename in filenames:

                file_path =os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                df = pd.read_csv(file_path)
                df_target_vals = list(df['Category'].unique())

                df_target_vals.sort()

                if schema_target_vals==df_target_vals:
                    valid_filenames.append(filename)

            if valid_filenames:
                cont = True
                message = "There are files with valid columns"
            else:
                cont = False
                message = "There are no files with valid columns"

            return (cont, message, valid_filenames)

        except Exception as e:
            raise NASException(e, sys) from e

    def combine(self, filenames: List[str]) -> Tuple[bool, str, str, str, str]:
        try:
            list_dataframes = []

            for filename in filenames:
                file_path = os.path.join(self.data_ingestion_artifact.ingested_dir, filename)
                df = pd.read_csv(file_path)
                list_dataframes.append(df)

            final_dataframe = pd.concat(list_dataframes, ignore_index=True)

            schema_input_columns = self.schema_dict['ColumnNames']
            schema_target_column = self.schema_dict['TargetColumn'] 

            schema_columns = schema_input_columns

            if self.data_validation_config.validated_prediction_dir is None:
                
                validated_train_file_path = os.path.join(self.data_validation_config.validated_train_dir, "train_dataset.csv")
                validated_prediction_file_path = None
                
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
            else:
                
                validated_train_file_path = None
                validated_prediction_file_path = os.path.join(self.data_validation_config.validated_prediction_dir, "prediction_dataset.csv")

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

            return (is_successful, message, validated_train_file_path, validated_prediction_file_path)

        except Exception as e:
            raise NASException(e, sys) from e


    def initiate_validation(self) -> DataValidationArtifact:
        try:
            
            regex = self.regex_file_name()
            cont, message, filenames = self.check_valid_filename(regex=regex)

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

            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact
            
            cont, message, filenames = self.check_correct_datatypes(filenames=filenames)

            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact

            cont, message, filenames = self.check_correct_domain_values(filenames=filenames)

            if not cont:
                data_validation_artifact = DataValidationArtifact(
                    is_validated=cont,
                    message=message,
                    train_file_path=None,
                    prediction_file_path=None
                )
                return data_validation_artifact

            is_validated, message, train_file_path,  prediction_file_path = self.combine(
                filenames=filenames)

            data_validation_artifact = DataValidationArtifact(
                is_validated=is_validated,
                message=message,
                train_file_path=train_file_path,
                prediction_file_path=prediction_file_path
            )
            return data_validation_artifact

        except Exception as e:
            raise NASException(e, sys) from e
