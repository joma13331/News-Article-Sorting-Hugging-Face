import sys

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASDataIngestion import NASDataIngestion
from NewsArticleSorting.NASComponents.NASDataValidation import NASDataValidation
from NewsArticleSorting.NASComponents.NASDataPreprocessing import NASDataPreprocessing
from NewsArticleSorting.NASComponents.NASModelTraining import NASModelTraining
from NewsArticleSorting.NASComponents.NASModelEvaluation import NASModelEvaluation
from NewsArticleSorting.NASComponents.NASModelPusher import NASModelPusher


class NASTrainingPipeline:
    """
    Class Name: NASTrainingPipeline
    Description: Includes all the methods that are needed train the model for News Article Sorting.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self) -> None:
        logging.info( f"{'*'*20} Training Pipeline log started {'*'*20}")

        self.nas_config = NASConfiguration()
        self.data_ingestion_config = self.nas_config.get_data_ingestion_config()
        self.data_validation_config = self.nas_config.get_data_validation_config()
        self.data_preprocessing_config = self.nas_config.get_data_preprocessing_config()
        self.model_training_config = self.nas_config.get_model_training_config()
        self.model_evaluation_config = self.nas_config.get_model_evaluation_config()
        self.model_pusher_config = self.nas_config.get_model_pusher_config()

    def complete_training_pipeline(self)-> str:
        """
        Method Name: complete_training_pipeline

        Description: This method combines all the components that are used to train a NLP model

        returns: str: a message regarding training completion.
        """
        try:
            # Data Ingestion
            data_injector = NASDataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_injection_artifact = data_injector.initiate_data_ingestion()
            if not data_injection_artifact.is_ingested:
                return data_injection_artifact.message

            # Data Validation
            data_validator = NASDataValidation(data_ingestion_artifact=data_injection_artifact,
                                                data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validator.initiate_validation()
            if not data_validation_artifact.is_validated:
                return data_validation_artifact.message

            # Data Preprocessing
            data_preprocessor = NASDataPreprocessing(
                data_validation_artifact=data_validation_artifact,
                data_preprocessing_config=self.data_preprocessing_config
            )
            data_preprocessing_artifact = data_preprocessor.initiate_data_preprocessing()
            if not data_preprocessing_artifact.is_preprocessed:
                return data_preprocessing_artifact.message

            # Model Training
            model_trainer = NASModelTraining(data_preprocessing_artifact=data_preprocessing_artifact,
                                model_training_config=self.model_training_config)
            model_trainer_artifact = model_trainer.initiate_model_training()
            if not model_trainer_artifact.is_trained:
                return model_trainer_artifact.message

            # Model Evaluation
            model_evaluator = NASModelEvaluation(
                model_training_config=self.model_training_config,
                model_evaluation_config=self.model_evaluation_config,
                data_preprocessing_artifact=data_preprocessing_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluator.initiate_evaluation()
            if not model_evaluation_artifact.is_accepted:
                return model_evaluation_artifact.message

            # Model Pushing
            model_pusher = NASModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact
            )
            model_pusher_artifact = model_pusher.initiate_model_push()

            return model_pusher_artifact.message

        except Exception as e:
            raise NASException(e, sys) from e

if __name__ == "__main__":
    training_pipeline = NASTrainingPipeline()
    print(training_pipeline.complete_training_pipeline())