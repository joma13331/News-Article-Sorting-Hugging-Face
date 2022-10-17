import sys
import numpy as np
import pandas as pd
from typing import List

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

from NewsArticleSorting.NASEntity.NASConfigEntity import DataPreprocessingConfig, ModelPusherConfig, ModelTrainingConfig, PredictorConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import PredictorArtifact
from NewsArticleSorting.NASEntity.NASModels import NASModel
from NewsArticleSorting.NASUtils.utils import read_yaml_file,load_model


class NASPredictor:
    """
    Class Name: NASPredictor
    Description: Includes all the methods that are needed to make prediction on the data passed.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self,
                data_preprocessing_config: DataPreprocessingConfig,
                predictor_config: PredictorConfig,
                model_pusher_config: ModelPusherConfig,
                model_training_config: ModelTrainingConfig
                ) -> None:
        try:
            logging.info( f"{'*'*20} Model Prediction log started {'*'*20}")

            self.data_preprocessing_config = data_preprocessing_config
            self.predictor_config = predictor_config
            self.model_pusher_config = model_pusher_config
            self.model_training_config = model_training_config

            model_info = read_yaml_file(self.model_pusher_config.deployed_model_details_path)
            self.model_name = model_info["model_name"]

        except Exception as e:
            raise NASException(e, sys) from e

    def load_from_disk(self):
        """
        Method Name: load_from_disk

        Description: This method loads the DatasetDict object from  the disk

        returns: DatasetDict - the data to predict through the model on.
        """
        try:
            # obtaining the data in the DatasetSict format from the disk
            nas_dataset = load_from_disk(self.data_preprocessing_config.preprocessed_pred_dir)
            logging.info(f"The dataset is loaded from {self.data_preprocessing_config.preprocessed_pred_dir}")

            return nas_dataset
        except Exception as e:
            raise NASException(e, sys) from e 

    def tokenizer_function(self, example):
        # Tokenizing the text
        return self.tokenizer(example[self.model_training_config.input_feature],
                padding=self.model_training_config.padding_type,
                truncation=self.model_training_config.truncation,
                max_length=self.model_training_config.max_seq_length,
                )

    def prediction(self, model, data_loader, device)-> List[List[float]]:
        """
        Method Name: prediction
        parameter: data_loader - an object of pytorch's DataLoader class which helps in loading data to the pytorch model
                   model - pytorch model which will perform classification
                   device - where prediction on the data is to be done(cpu/gpu)


        Description: This method predicts,using the NLP model, on the data passed through dataloader
        """
        try:
            # Setting the model for evaluation
            model.eval()

            fin_outputs = []
            
            # Ensuring no gradients are calculated during evaluation.
            with torch.no_grad():
                # Iterating through all the batches in the dataset
                for index, data in tqdm(enumerate(data_loader), total=len(data_loader)):

                    # obtaining the inputs from the tokenized text and sending then to the device selected for training
                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]

                    input_ids = input_ids.to(device, dtype=torch.long)
                    attention_mask = attention_mask.to(device, dtype=torch.long)

                    # Obtaining the ouput from tihe model
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

            return fin_outputs

        except Exception as e:
            raise NASException(e, sys) from e

    
    def initiate_prediction(self)-> PredictorArtifact:
        """
        Method Name: initiate_prediction

        Description: This method Combines all the relevant methods to achieve prediction on the data passed. 

        returns: PredictorArtifact - contains all the relevant information obtained after prediction.
        """
        try:
            # Setting the device based on whether GPU for DL evaluation is available or not   
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Model name for model, tokenizer that will be selected for training
            model = NASModel(model_name=self.model_name)
            model.load_state_dict(torch.load(self.model_pusher_config.deployed_model_path))
            model.to(device)

            nas_dataset = self.load_from_disk()
            tokenized_dataset = nas_dataset.map(self.tokenizer_function, batched=True, remove_columns='Text')

            pred_dataloader = DataLoader(
                tokenized_dataset['train'],
                batch_size=self.predictor_config.pred_batch_size,
                collate_fn=data_collator
            )

            fin_outputs = self.prediction(model=model,
                                data_loader=pred_dataloader,
                                device=device)
            fin_outputs = np.argmax(np.array(fin_outputs), axis=1)

            ohe = load_model(self.data_preprocessing_config.ohe_file_path)

            # Obtaining Category from One hot encoded output 
            fin_outputs = [list(ohe.categories_[0])[ind] for ind in fin_outputs]

            prediction_result = pd.DataFrame()
            prediction_result['ArticleId'] = nas_dataset['train']['ArticleId'][:]
            prediction_result['Text'] = nas_dataset['train']['Text'][:]
            prediction_result["Category"] = fin_outputs

            # Storing results to a csv file
            prediction_result.to_csv(self.predictor_config.prediction_result_filepath, sep=" ")
            
            message = f"The predictions by the model have been completed and the results are save at {self.predictor_config.prediction_result_filepath}"
            logging.info(message)

            predictor_artifact = PredictorArtifact(prediction_result_path=self.predictor_config.prediction_result_filepath, message=message)
            logging.info(f"Predictor Artifact: {predictor_artifact}")
            
            return predictor_artifact


        except Exception as e:
            raise NASException(e, sys) from e

    def initiate_prediction_single_sentence(self, sentence)-> str:
        """
        Method Name: initiate_prediction

        Description: This method performs prediction on a single sentence. 

        returns: str - Category of the text on which prediction is done.
        """
        try:

            # Setting the device based on whether GPU for DL training is available or not   
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            review = str(sentence)

            tokenized_inputs = self.tokenizer(review,
                padding=self.model_training_config.padding_type,
                truncation=self.model_training_config.truncation,
                max_length=self.model_training_config.max_seq_length,
                )

            # obtaining the inputs from the tokenized text and sending then to the device selected for training
            ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long).unsqueeze(0)
            mask = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.long).unsqueeze(0)

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            # Model name for model, tokenizer that will be selected for training
            model = NASModel(model_name=self.model_name)
            model.load_state_dict(torch.load(self.model_pusher_config.deployed_model_path))
            model.to(device)

            outputs = model(
                input_ids=ids,
                attention_mask=mask
            )

            output = torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist()
            output = np.argmax(np.array(output), axis=1)

            ohe = load_model(self.data_preprocessing_config.ohe_file_path)
            # Obtaining Category from One hot encoded output 
            final_output = [list(ohe.categories_[0])[ind] for ind in list(output)]

            logging.info(f"prediction on sentence {sentence} done with classification: {final_output[0]}")

            return final_output[0]


        except Exception as e:
            raise NASException(e, sys) from e