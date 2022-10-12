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
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataPreprocessingArtifact, PredictorArtifact, ModelTrainerArtifact
from NewsArticleSorting.NASEntity.NASModels import NASModel
from NewsArticleSorting.NASUtils.utils import read_yaml_file,load_model


class NASPredictor:


    def __init__(self,
                data_preprocessing_config: DataPreprocessingConfig,
                predictor_config: PredictorConfig,
                model_pusher_config: ModelPusherConfig,
                model_training_config: ModelTrainingConfig
                ) -> None:
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.predictor_config = predictor_config
            self.model_pusher_config = model_pusher_config
            self.model_training_config = model_training_config

            model_info = read_yaml_file(self.model_pusher_config.deployed_model_details_path)
            self.model_name = model_info["model_name"]

        except Exception as e:
            raise NASException(e, sys) from e

    def load_from_disk(self):
        try:
            nas_dataset = load_from_disk(self.data_preprocessing_config.preprocessed_pred_dir)
            return nas_dataset
        except Exception as e:
            raise NASException(e, sys) from e 

    def tokenizer_function(self, example):
        return self.tokenizer(example[self.model_training_config.input_feature],
                padding=self.model_training_config.padding_type,
                truncation=self.model_training_config.truncation,
                max_length=self.model_training_config.max_seq_length,
                )

    def prediction(self, model, data_loader, device)-> List[List[float]]:
        try:
            model.eval()

            fin_outputs = []

            with torch.no_grad():
                for index, data in tqdm(enumerate(data_loader), total=len(data_loader)):

                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]

                    input_ids = input_ids.to(device, dtype=torch.long)
                    attention_mask = attention_mask.to(device, dtype=torch.long)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

            return fin_outputs

        except Exception as e:
            raise NASException(e, sys) from e

    
    def initiate_prediction(self)-> PredictorArtifact:
        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

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
            fin_outputs = [list(ohe.categories_[0])[ind] for ind in fin_outputs]

            prediction_result = pd.DataFrame()
            prediction_result['ArticleId'] = nas_dataset['train']['ArticleId'][:]
            prediction_result['Text'] = nas_dataset['train']['Text'][:]
            prediction_result["Category"] = fin_outputs

            prediction_result.to_csv(self.predictor_config.prediction_result_filepath, sep=" ")
            
            is_prediction_done = True
            message = f"The predictions by the model have been completed and the results are save at {self.predictor_config.prediction_result_filepath}"

            return PredictorArtifact(is_prediction_done=is_prediction_done, message=message)


        except Exception as e:
            raise NASException(e, sys) from e