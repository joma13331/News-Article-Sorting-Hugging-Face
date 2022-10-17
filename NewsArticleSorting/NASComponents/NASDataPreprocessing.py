import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from datasets import DatasetDict, load_dataset


from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASUtils.utils import load_model, save_model

from NewsArticleSorting.NASEntity.NASConfigEntity import DataPreprocessingConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataPreprocessingArtifact, DataValidationArtifact


class NASDataPreprocessing:
    """
    Class Name: NASDataPreprocessing
    Description: Includes all the methods that are needed to prerocess the data so that best possible model maybe created.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self, 
        data_preprocessing_config: DataPreprocessingConfig,
        data_validation_artifact:DataValidationArtifact) -> None:
        
        try:
            logging.info( f"{'*'*20} Data Preprocessing log started {'*'*20}")

            self.data_preprocessing_config = data_preprocessing_config
            self.data_validation_artifact = data_validation_artifact
            self.ohe = OneHotEncoder(sparse=False)
            self.original_columns = []

        except Exception as e:
            raise NASException(e, sys) from e

    def obtain_train_test_val_dataset_dict(self)-> DatasetDict:
        """
        Method Name: obtain_train_test_val_dataset_dict
        Description: This method return the DatasetDict created from the validated data which contains splits train, validation,
                    test.

        return: DatasetDict: Dataset format which will be used further to train models
        """
        try:
            # Checking to see if we are performing prediction or not
            if self.data_preprocessing_config.preprocessed_pred_dir is None:

                # Using dataset(transformers) library to load the files as a DatasetDict and creating train, val, test splits        
                nas_dataset = load_dataset("csv", data_files=self.data_validation_artifact.train_file_path)           
                nas_dataset = nas_dataset['train'].train_test_split(test_size=0.2, seed=42)
                test_dataset = nas_dataset['test'].train_test_split(test_size=0.5, seed=42)
                nas_dataset['validation'] = test_dataset.pop('train')
                nas_dataset['test'] = test_dataset.pop('test')

                # Fitting a One Hot Encoder model and transforming the data
                self.ohe.fit_transform(np.array(nas_dataset['train']['Category']).reshape(-1,1))
                # Saving the One Hot Encoder model for future use
                save_model(model=self.ohe, model_file_path=self.data_preprocessing_config.ohe_file_path)
                logging.info(f"the validated dataset has been divided into train, validation and test splits")
            
            else:
                # Loading the file to predict on as a DatasetDict object 
                nas_dataset = load_dataset("csv", data_files=self.data_validation_artifact.prediction_file_path)
                # Loading an already fitted One Hot Encoder model
                self.ohe = load_model(self.data_preprocessing_config.ohe_file_path)
                logging.info(f"the validated dataset has been loaded to perform prediction on")
            
            self.original_columns = nas_dataset['train'].column_names
            self.original_columns.remove('Text')

            return nas_dataset

        except Exception as e:
            raise NASException(e, sys) from e

    
    def labels_to_one_hot_encoded(self, example: DatasetDict)-> dict:
        """
        Method Name: labels_to_one_hot_encoded
        parameter: example-a DatasetDict record which will be passed during the map method of a DatasetDict object.

        Description: This method one hot encodes the Category/label column.

        return: dict: dictionary with 'labels' as key and one hot encoded output as 'value 
        """
        try:
            # One hot encoding the models
            return {'labels': [tuple(x) for x in self.ohe.transform(np.array(example['Category']).reshape(-1,1))]}

        except Exception as e:
            raise NASException(e, sys) from e


    def obtain_one_hot_encoded_labels(self, dataset_dict: DatasetDict)-> DatasetDict:
        """
        Method Name: obtain_one_hot_encoded_labels
        parameter: dataset_dict-the DatasetDict object onto which method labels_to_one_hot_encoded is to be mapped

        Description: This method maps the method 'labels_to_one_hot_encoded' to all the records in the dataset.

        return: DatasetDict: Dataset format which will be used further to train models
        """
        try:
            # Converting the category column in one hot encoded label
            dataset_dict = dataset_dict.map(self.labels_to_one_hot_encoded, remove_columns=self.original_columns, batched=True)
            logging.info("one hot encoded labels have been obtained")
            return dataset_dict

        except Exception as e:
            raise NASException(e, sys) from e

    
    def save_data_to_disk(self, dataset_dict:DatasetDict)-> None:
        """
        Method Name: save_data_to_disk
        parameter: dataset_dict-the DatasetDict object  which is to be aved to disk

        Description: This method saves the DatasetDict object ont the disk
        """
        try:
            # Saving the pre-processed data at the appropriate loaction
            if self.data_preprocessing_config.preprocessed_pred_dir is None:
                dataset_dict.save_to_disk(self.data_preprocessing_config.preprocessed_train_dir)
                logging.info(f"The dataset with the train, validation, test split has been saved to {self.data_preprocessing_config.preprocessed_train_dir}")
            else: 
                dataset_dict.save_to_disk(self.data_preprocessing_config.preprocessed_pred_dir)
                logging.info(f"The prediction dataset has been saved to {self.data_preprocessing_config.preprocessed_pred_dir}")

        
        except Exception as e:
            raise NASException(e, sys) from e
    
    def initiate_data_preprocessing(self)-> DataPreprocessingArtifact:
        """
        Method Name: initiate_data_preprocessing
        Description: Combines all the relevant methods to achieve data preprocessing.

        return: Data Preprocessing Artifact: contains all the relevant information for model training
        """
        try:
            # Obtaiing the splits in the dataset 
            nas_dataset = self.obtain_train_test_val_dataset_dict()
            
            # Encoding the outputs
            if self.data_preprocessing_config.preprocessed_pred_dir is None:
                nas_dataset = self.obtain_one_hot_encoded_labels(dataset_dict=nas_dataset)
                message = f"The dataset is saved at {self.data_preprocessing_config.preprocessed_train_dir}"
            else:
                message = f"The dataset is saved at {self.data_preprocessing_config.preprocessed_pred_dir}"

            # Saving the preprocessed data
            self.save_data_to_disk(nas_dataset)

            is_preprocessed = True

            data_preprocessing_artifact = DataPreprocessingArtifact(
                is_preprocessed=is_preprocessed,
                message=message,
                train_dir_path=self.data_preprocessing_config.preprocessed_train_dir,
                pred_dir_path=self.data_preprocessing_config.preprocessed_pred_dir
            )
            
            logging.info(f"The Data Preprocessing Artifact: {data_preprocessing_artifact}")

            return data_preprocessing_artifact

        except Exception as e:
            raise NASException(e, sys) from e