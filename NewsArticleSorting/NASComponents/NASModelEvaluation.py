import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASUtils.utils import write_yaml_file

from NewsArticleSorting.NASEntity.NASConfigEntity import ModelEvaluationConfig, ModelTrainingConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import DataPreprocessingArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from NewsArticleSorting.NASEntity.NASModels import NASModel


class NASModelEvaluation:

    def __init__(self,
    model_evaluation_config: ModelEvaluationConfig,
    model_training_config: ModelTrainingConfig,
    data_preprocessing_artifact: DataPreprocessingArtifact,
    model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            self.model_training_config = model_training_config
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_preprocessing_artifact = data_preprocessing_artifact

        except Exception as e:
            raise NASException(e, sys) from e

    def load_from_disk(self):
        try:
            nas_dataset = load_from_disk(self.data_preprocessing_artifact.train_dir_path)
            return nas_dataset
        except Exception as e:
            raise NASException(e, sys) from e
    
    def tokenizer_function(self, example):
        return self.tokenizer(example[self.model_training_config.input_feature],
                padding=self.model_training_config.padding_type,
                truncation=self.model_training_config.truncation,
                max_length=self.model_training_config.max_seq_length,
                )


    def eval_fn(self, data_loader, model, device):

        model.eval()

        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                labels = data["labels"]

                input_ids = input_ids.to(device, dtype=torch.long)
                attention_mask = attention_mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                fin_targets.extend(labels.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

        return fin_outputs, fin_targets

    def initiate_evaluation(self) -> ModelEvaluationArtifact:
        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            model = NASModel(model_name=self.model_trainer_artifact.model_name)
            model.to(device)

            model.load_state_dict(torch.load(self.model_trainer_artifact.trained_model_path))

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_trainer_artifact.model_name)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            nas_dataset = self.load_from_disk()

            nas_dataset = nas_dataset.map(self.tokenizer_function, batched=True, remove_columns=['Text'])

            test_data_loader = DataLoader(
                nas_dataset['test'],
                batch_size=self.model_evaluation_config.eval_batch_size,
                collate_fn=data_collator
            )

            outputs, targets = self.eval_fn(test_data_loader, model, device)

            outputs = np.array(outputs)>=np.max(outputs,axis=1).reshape(-1,1)
            accuracy = metrics.accuracy_score(targets, outputs)
            
            print(f"Accuracy Score = {accuracy}")

            if accuracy > self.model_evaluation_config.base_accuracy:
                is_accepted = True
                message = f"The model trained at  {self.model_trainer_artifact.trained_model_path} has test accuracy of {accuracy}"\
                    f" and is accepted"
            else:
                is_accepted = False
                message = f"The model trained at  {self.model_trainer_artifact.trained_model_path} has test accuracy of {accuracy}"\
                    f" and is not accepted"

            eval_result_dict = {
                "model_name": self.model_trainer_artifact.model_name,
                "optimizer": self.model_trainer_artifact.optimizer,
                "lr": self.model_trainer_artifact.lr,
                "test_accuracy": float(accuracy)
            }

            write_yaml_file(file_path=self.model_evaluation_config.model_evaluation_result_path,
                            data=eval_result_dict)

            return ModelEvaluationArtifact(
                is_accepted=is_accepted,
                message=message,
                model_evaluation_result_path=self.model_evaluation_config.model_evaluation_result_path
            )

        except Exception as e:
            raise NASException(e, sys) from e
