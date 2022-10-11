import os,shutil
import sys
from typing import Tuple

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASUtils.utils import read_yaml_file

from NewsArticleSorting.NASEntity.NASConfigEntity import  ModelPusherConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact


class NASModelPusher:

    def __init__(self,
    model_pusher_config: ModelPusherConfig,
    model_evaluation_artifact: ModelEvaluationArtifact,
    model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise NASException(e, sys) from e

    def copy_evaluated_model_if_better(self)-> Tuple[bool, str]:
        try:

            is_pushed = False
            message = "Trained model not better than deployed model hence not pushed"

            if not os.path.isfile(self.model_pusher_config.deployed_model_path):
                shutil.copyfile(
                    src=self.model_trainer_artifact.trained_model_path,
                    dst= self.model_pusher_config.deployed_model_path)

                shutil.copyfile(
                    src=self.model_evaluation_artifact.model_evaluation_result_path,
                    dst=self.model_pusher_config.deployed_model_details_path
                )

                is_pushed = True
                message = "No previous model was trained hence Trained model is pushed"

            else:
                deployed_model_results = read_yaml_file(
                    file_path=self.model_pusher_config.deployed_model_details_path)

                evaluated_model_results = read_yaml_file(
                    file_path=self.model_evaluation_artifact.model_evaluation_result_path
                )

                if evaluated_model_results["test_accuracy"] > deployed_model_results["test_accuracy"]:
                    shutil.copyfile(
                    src=self.model_trainer_artifact.trained_model_path,
                    dst= self.model_pusher_config.deployed_model_path)

                    shutil.copyfile(
                        src=self.model_evaluation_artifact.model_evaluation_result_path,
                        dst=self.model_pusher_config.deployed_model_details_path
                    )
                    is_pushed = True
                    message = f"Trained model (accuracy = {evaluated_model_results['test_accuracy']}) better than"\
                    f"deployed model (accuracy = {deployed_model_results['test_accuracy']}) hence model pushed"


            return is_pushed, message    
        except Exception as e:
            raise NASException(e, sys) from e


    def initiate_model_push(self) -> ModelPusherArtifact:
        try:
            is_pushed, message = self.copy_evaluated_model_if_better()

            return ModelPusherArtifact(
                is_pushed=is_pushed,
                message=message
            )

        except Exception as e:
            raise NASException(e, sys) from e
