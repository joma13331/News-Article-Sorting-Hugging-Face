import os
import sys
import pickle
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASUtils.utils import load_model, save_model

from NewsArticleSorting.NASEntity.NASConfigEntity import DataPreprocessingConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataPreprocessingArtifact, DataValidationArtifact


class NASDataPreprocessing:

    def __init__(self, 
        data_preprocessing_config: DataPreprocessingConfig,
        data_validation_artifact:DataValidationArtifact) -> None:
        
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.data_validation_artifact = data_validation_artifact
            self.ohe = OneHotEncoder(sparse=False)
            self.original_columns = []

        except Exception as e:
            raise NASException(e, sys) from e

    def obtain_train_test_val_dataset_dict(self)-> DatasetDict:
        try:
            if self.data_preprocessing_config.preprocessed_pred_dir is None:

                nas_dataset = load_dataset("csv", data_files=self.data_validation_artifact.train_file_path)           
                nas_dataset = nas_dataset['train'].train_test_split(test_size=0.2, seed=42)
                test_dataset = nas_dataset['test'].train_test_split(test_size=0.5, seed=42)
                nas_dataset['validation'] = test_dataset.pop('train')
                nas_dataset['test'] = test_dataset.pop('test')

                self.ohe.fit_transform(np.array(nas_dataset['train']['Category']).reshape(-1,1))
                save_model(model=self.ohe, model_file_path=self.data_preprocessing_config.ohe_file_path)
            
            else: 
                nas_dataset = load_dataset("csv", data_files=self.data_validation_artifact.prediction_file_path)
                self.ohe = load_model(self.data_preprocessing_config.ohe_file_path)
                

            self.original_columns = nas_dataset['train'].column_names
            self.original_columns.remove('Text')

            return nas_dataset

        except Exception as e:
            raise NASException(e, sys) from e

    
    def labels_to_one_hot_encoded(self, example: DatasetDict)-> dict:
        try:
            return {'labels': [tuple(x) for x in self.ohe.transform(np.array(example['Category']).reshape(-1,1))]}

        except Exception as e:
            raise NASException(e, sys) from e

    def obtain_one_hot_encoded_labels(self, dataset_dict: DatasetDict)-> DatasetDict:
        try:
            dataset_dict = dataset_dict.map(self.labels_to_one_hot_encoded, remove_columns=self.original_columns, batched=True)
            return dataset_dict

        except Exception as e:
            raise NASException(e, sys) from e

    
    def save_data_to_disk(self, dataset_dict:DatasetDict)-> None:
        try:
            if self.data_preprocessing_config.preprocessed_pred_dir is None:
                dataset_dict.save_to_disk(self.data_preprocessing_config.preprocessed_train_dir)
            else: 
                dataset_dict.save_to_disk(self.data_preprocessing_config.preprocessed_pred_dir)
        
        except Exception as e:
            raise NASException(e, sys) from e
    
    def initiate_data_preprocessing(self)-> DataPreprocessingArtifact:
        try:
            nas_dataset = self.obtain_train_test_val_dataset_dict()
            nas_dataset = self.obtain_one_hot_encoded_labels(dataset_dict=nas_dataset)
            self.save_data_to_disk(nas_dataset)

            is_preprocessed = True

            if self.data_preprocessing_config.preprocessed_pred_dir is None:
                message = f"The dataset is saved at {self.data_preprocessing_config.preprocessed_train_dir}"
            else:
                message = f"The dataset is saved at {self.data_preprocessing_config.preprocessed_pred_dir}"


            return DataPreprocessingArtifact(
                is_preprocessed=is_preprocessed,
                message=message,
                train_dir_path=self.data_preprocessing_config.preprocessed_train_dir,
                pred_dir_path=self.data_preprocessing_config.preprocessed_pred_dir
            )
        except Exception as e:
            raise NASException(e, sys) from e