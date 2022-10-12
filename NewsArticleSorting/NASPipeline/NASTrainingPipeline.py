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

    def __init__(self) -> None:
        self.nas_config = NASConfiguration()
        self.data_ingestion_config = self.nas_config.get_data_ingestion_config()
        self.data_validation_config = self.nas_config.get_data_validation_config()
        self.data_preprocessing_config = self.nas_config.get_data_preprocessing_config()
        self.model_training_config = self.nas_config.get_model_training_config()
        self.model_evaluation_config = self.nas_config.get_model_evaluation_config()
        self.model_pusher_config = self.nas_config.get_model_pusher_config()

    def complete_training_pipeline(self)-> str:
        try:

            data_injector = NASDataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_injection_artifact = data_injector.initiate_data_ingestion()
            if not data_injection_artifact.is_ingested:
                return data_injection_artifact.message

            data_validator = NASDataValidation(data_ingestion_artifact=data_injection_artifact,
                                                data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validator.initiate_validation()
            if not data_validation_artifact.is_validated:
                return data_validation_artifact.message

            data_preprocessor = NASDataPreprocessing(
                data_validation_artifact=data_validation_artifact,
                data_preprocessing_config=self.data_preprocessing_config
            )
            data_preprocessing_artifact = data_preprocessor.initiate_data_preprocessing()
            if not data_preprocessing_artifact.is_preprocessed:
                return data_preprocessing_artifact.message

            model_trainer = NASModelTraining(data_preprocessing_artifact=data_preprocessing_artifact,
                                model_training_config=self.model_training_config)
            model_trainer_artifact = model_trainer.initiate_model_training()
            if not model_trainer_artifact.is_trained:
                return model_trainer_artifact.message

            model_evaluator = NASModelEvaluation(
                model_training_config=self.model_training_config,
                model_evaluation_config=self.model_evaluation_config,
                data_preprocessing_artifact=data_preprocessing_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluator.initiate_evaluation()
            if not model_evaluation_artifact.is_accepted:
                return model_evaluation_artifact.message

            model_pusher = NASModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact
            )
            model_pusher_artifact = model_pusher.initiate_model_push()

            return model_pusher_artifact.message

        except Exception as e:
            raise NASException(e, sys) from e

    def initiate_training_pipeline(self)-> str:
        try:

           message = self.complete_training_pipeline()
           return message
        except Exception as e:
            raise NASException(e, sys) from e

if __name__ == "__main__":
    training_pipeline = NASTrainingPipeline()
    print(training_pipeline.initiate_training_pipeline())