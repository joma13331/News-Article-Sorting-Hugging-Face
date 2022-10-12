import sys

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASDataIngestion import NASDataIngestion
from NewsArticleSorting.NASComponents.NASDataValidation import NASDataValidation
from NewsArticleSorting.NASComponents.NASDataPreprocessing import NASDataPreprocessing
from NewsArticleSorting.NASComponents.NASPredictor import NASPredictor

class NASPredictionPipeline:

    def __init__(self)-> None:
        try:
            self.nas_config = NASConfiguration(is_training=False)
            self.data_ingestion_config = self.nas_config.get_data_ingestion_config()
            self.data_validation_config = self.nas_config.get_data_validation_config()
            self.data_preprocessing_config = self.nas_config.get_data_preprocessing_config()
            self.predictor_config = self.nas_config.get_predictor_config()

            self.model_pusher_config = self.nas_config.get_model_pusher_config()
            self.model_training_config = self.nas_config.get_model_training_config()

        except Exception as e:
            raise NASException(e, sys) from e

    def complete_prediction_pipeline(self)-> str:
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

            predictor = NASPredictor(
                data_preprocessing_config=self.data_preprocessing_config,
                predictor_config=self.predictor_config,
                model_pusher_config=self.model_pusher_config,
                model_training_config=self.model_training_config
            )

            predictor_artifact = predictor.initiate_prediction()

            return predictor_artifact.message

        except Exception as e:
            raise NASException(e, sys) from e

    def initiate_prediction_pipeline(self)-> str:
        try:

           message = self.complete_prediction_pipeline()
           return message
           
        except Exception as e:
            raise NASException(e, sys) from e


if __name__ == "__main__":
    prediction_pipeline = NASPredictionPipeline()
    print(prediction_pipeline.initiate_prediction_pipeline())