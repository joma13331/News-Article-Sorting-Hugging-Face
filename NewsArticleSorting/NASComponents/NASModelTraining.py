import numpy as np
import optuna
from optuna.pruners import ThresholdPruner
from tqdm import tqdm
import sys
import torch
from sklearn import metrics
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


from transformers import get_linear_schedule_with_warmup,AutoTokenizer, DataCollatorWithPadding

from datasets import load_from_disk

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

from NewsArticleSorting.NASEntity.NASArtifactEntity import DataPreprocessingArtifact, ModelTrainerArtifact
from NewsArticleSorting.NASEntity.NASConfigEntity import ModelTrainingConfig
from NewsArticleSorting.NASEntity.NASModels import NASModel


class NASModelTraining:
    

    def __init__(self, 
                data_preprocessing_artifact:DataPreprocessingArtifact,
                model_training_config: ModelTrainingConfig) -> None:
        try:
            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.model_training_config = model_training_config
            

        except Exception as e:
            raise NASException(e, sys) from e
    

    def tokenizer_function(self, example):
        return self.tokenizer(example[self.model_training_config.input_feature],
                padding=self.model_training_config.padding_type,
                truncation=self.model_training_config.truncation,
                max_length=self.model_training_config.max_seq_length,
                )

    
    def loss_fn(self, outputs, targets):

        softmax_output = torch.softmax(outputs, dim=1)
        log_softmaxed_output = torch.log(softmax_output)
        multiplied_result = torch.mul(log_softmaxed_output, targets)

        loss = torch.sum(multiplied_result)*(-1)

        return loss

    
    def train_fn(self, data_loader, model, optimizer, device, scheduler):

        model.train()

        for index, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = self.loss_fn(outputs=outputs, targets=labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

    
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


    def load_from_disk(self):
        try:
            nas_dataset = load_from_disk(self.data_preprocessing_artifact.train_dir_path)
            return nas_dataset
        except Exception as e:
            raise NASException(e, sys) from e   


    def objective(self, trial)-> float:
           
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model_name = trial.suggest_categorical("model_name", self.model_training_config.models)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        model = NASModel(model_name=model_name)
        model.to(device)

        param_optimizer = list(model.named_parameters()) 
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_name = trial.suggest_categorical("optimizer_name", self.model_training_config.optimizers)
        lr = trial.suggest_float("lr", float(self.model_training_config.learning_rate_start),
                                    float(self.model_training_config.learning_rate_end), log=True)

        optimizer = getattr(optim, optimizer_name)(optimizer_parameters, lr=lr)

        nas_dataset = self.load_from_disk()

        nas_dataset = nas_dataset.map(self.tokenizer_function, batched=True, remove_columns=['Text'])

        train_data_loader = DataLoader(
            nas_dataset['train'],
            batch_size = self.model_training_config.train_batch_size,
            collate_fn=data_collator
        )

        val_data_loader = DataLoader(
            nas_dataset['validation'],
            batch_size=self.model_training_config.train_batch_size,
            collate_fn=data_collator
        )

        num_train_steps=int(len(train_data_loader)*self.model_training_config.hyperparameter_tuning_epochs)

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=num_train_steps
            )

        
        for epoch in range(self.model_training_config.hyperparameter_tuning_epochs):
            self.train_fn(train_data_loader, model, optimizer, device, scheduler)
            outputs, targets = self.eval_fn(val_data_loader, model, device)
            outputs = np.array(outputs)>=np.max(outputs,axis=1).reshape(-1,1)
            accuracy = metrics.accuracy_score(targets, outputs)
            print(f"Accuracy Score = {accuracy}")
        
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return accuracy    

    def train_best_model(self, model_name: str, optimizer_name: str, lr: float)-> float:
        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = NASModel(model_name=model_name)
            model.to(device)

            param_optimizer = list(model.named_parameters()) 
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

            optimizer_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.001,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = getattr(optim, optimizer_name)(optimizer_parameters, lr=lr)

            nas_dataset = self.load_from_disk()

            nas_dataset = nas_dataset.map(self.tokenizer_function, batched=True, remove_columns=['Text'])

            train_data_loader = DataLoader(
                nas_dataset['train'],
                batch_size = self.model_training_config.train_batch_size,
                collate_fn=data_collator
            )

            val_data_loader = DataLoader(
                nas_dataset['validation'],
                batch_size=self.model_training_config.train_batch_size,
                collate_fn=data_collator
            )

            num_train_steps=int(len(train_data_loader)*self.model_training_config.num_train_epochs)

            scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_train_steps
                )

            best_accuracy = 0
            for epoch in range(self.model_training_config.num_train_epochs):

                self.train_fn(train_data_loader, model, optimizer, device, scheduler)
                outputs, targets = self.eval_fn(val_data_loader, model, device)
                outputs = np.array(outputs)>=np.max(outputs,axis=1).reshape(-1,1)
                accuracy = metrics.accuracy_score(targets, outputs)
                
                print(f"Accuracy Score = {accuracy}")

                if accuracy > best_accuracy:
                    
                    if accuracy >= self.model_training_config.base_accuracy:
                        torch.save(model.state_dict(), self.model_training_config.trained_model_path)
                    best_accuracy = accuracy

                return best_accuracy

        except Exception as e:
            raise NASException(e,sys) from e

    def initiate_model_training(self)-> ModelTrainerArtifact:
        try:
            study = optuna.create_study(direction="maximize", pruner=ThresholdPruner(lower=0.2))
            study.optimize(lambda trial: self.objective(trial), n_trials=self.model_training_config.no_of_models_to_check)

            best_parameters = study.best_trial.params

            accuracy = self.train_best_model(model_name=best_parameters['model_name'],
                                    optimizer_name=best_parameters['optimizer_name'],
                                    lr=best_parameters['lr'])

            if accuracy >= self.model_training_config.base_accuracy:
                is_trained = True
                message = f" The relevant model is trained at {self.model_training_config.trained_model_path} with accuracy {accuracy}."
            else:
                is_trained = False
                message = f" The relevant model is not trained as accuracy was only {accuracy}."

            return ModelTrainerArtifact(
                is_trained=is_trained,
                message=message,
                trained_model_path=self.model_training_config.trained_model_path,
                model_name=best_parameters['model_name'],
                optimizer=best_parameters['optimizer_name'],
                lr=best_parameters['lr']
            )

        except Exception as e:
            raise NASException(e, sys) from e