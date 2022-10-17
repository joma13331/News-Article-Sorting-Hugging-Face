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
    """
    Class Name: NASModelEvaluation
    Description: Includes all the methods that are needed to create a NLP model to evaluate model that was trainined.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self,
    model_evaluation_config: ModelEvaluationConfig,
    model_training_config: ModelTrainingConfig,
    data_preprocessing_artifact: DataPreprocessingArtifact,
    model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            logging.info( f"{'*'*20}Model Evaluation log started {'*'*20}")

            self.model_training_config = model_training_config
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_preprocessing_artifact = data_preprocessing_artifact

        except Exception as e:
            raise NASException(e, sys) from e

    def load_from_disk(self):
        """
        Method Name: load_from_disk

        Description: This method loads the DatasetDict object from  the disk

        returns: DatasetDict - the data to train, validate and test the model on.
        """
        try:
            # obtaining the data in the DatasetSict format from the disk
            nas_dataset = load_from_disk(self.data_preprocessing_artifact.train_dir_path)
            logging.info(f"The dataset is loaded from {self.data_preprocessing_artifact.train_dir_path}")

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


    def eval_fn(self, data_loader, model, device):
        """
        Method Name: eval_fn
        parameter: data_loader - an object of pytorch's DataLoader class which helps in loading data to the pytorch model
                   model - pytorch model which will perform classification
                   device - where evaluation of model is to be done(cpu/gpu)


        Description: This method evaluates the NLP model based on the data passed through dataloader
        """
        try:
            # Setting the model for evaluation
            model.eval()

            fin_targets = []
            fin_outputs = []

            # Ensuring no gradients are calculated during evaluation.
            with torch.no_grad():
                # Iterating through all the batches in the dataset
                for index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                    
                    # obtaining the inputs from the tokenized text and sending then to the device selected for training
                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]
                    labels = data["labels"]

                    input_ids = input_ids.to(device, dtype=torch.long)
                    attention_mask = attention_mask.to(device, dtype=torch.long)
                    labels = labels.to(device, dtype=torch.long)

                    # Obtaining the ouput from tihe model
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    fin_targets.extend(labels.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

            return fin_outputs, fin_targets
        except Exception as e:
            raise NASException(e, sys) from e

    def initiate_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name: initiate_evaluation

        Description: This method Combines all the relevant methods to achieve model evaluation. 

        returns: ModelEvaluationArtifact - contains all the relevant information for further model pushing
        """
        try:
            # Setting the device based on whether GPU for DL evaluation is available or not   
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # Model name for model, tokenizer that will be selected for training
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
            # Obtaining accuracy of best trained model
            accuracy = metrics.accuracy_score(targets, outputs)
            
            print(f"Accuracy Score = {accuracy}")

            # Checking to see if model accuracy is above accepted base accuracy 
            if accuracy > self.model_evaluation_config.base_accuracy:
                is_accepted = True
                message = f"The model trained at  {self.model_trainer_artifact.trained_model_path} has test accuracy of {accuracy}"\
                    f" and is accepted"
            else:
                is_accepted = False
                message = f"The model trained at  {self.model_trainer_artifact.trained_model_path} has test accuracy of {accuracy}"\
                    f" and is not accepted"

            logging.info(message)

            # Storing in a python dictionary the details regarding the model that need to be saved
            eval_result_dict = {
                "model_name": self.model_trainer_artifact.model_name,
                "optimizer": self.model_trainer_artifact.optimizer,
                "lr": self.model_trainer_artifact.lr,
                "test_accuracy": float(accuracy)
            }

            # Storing the dictionary details into a yaml file
            write_yaml_file(file_path=self.model_evaluation_config.model_evaluation_result_path,
                            data=eval_result_dict)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_accepted=is_accepted,
                message=message,
                model_evaluation_result_path=self.model_evaluation_config.model_evaluation_result_path
            )
            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise NASException(e, sys) from e
