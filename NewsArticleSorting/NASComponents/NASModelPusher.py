import os,shutil
import sys
from typing import Tuple

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASUtils.utils import read_yaml_file

from NewsArticleSorting.NASEntity.NASConfigEntity import  ModelPusherConfig
from NewsArticleSorting.NASEntity.NASArtifactEntity import ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact


class NASModelPusher:
    """
    Class Name: NASModelPusher
    Description: Includes all the methods that are needed to make available the best trained model.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self,
    model_pusher_config: ModelPusherConfig,
    model_evaluation_artifact: ModelEvaluationArtifact,
    model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            logging.info( f"{'*'*20}Model Pushing log started {'*'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise NASException(e, sys) from e

    def copy_evaluated_model_if_better(self)-> Tuple[bool, str]:
        """
        Method Name: copy_evaluated_model_if_better

        Description: This method checks whether there is an existing model which is being server. If there is no
                     model being served or if the newly trained model is better that the existing model then the 
                     trained model is then overwritten for being served.

        return : bool: whether the model is pushed
                 str: message relevant to the action performed by this method
        """
        
        try:
            is_pushed = False
            message = "Trained model not better than deployed model hence not pushed"

            # Checking to see if any previous model is already being served
            if not os.path.isfile(self.model_pusher_config.deployed_model_path):
                # Copy the model and the evaluation details into the desired path
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
                # Reading the evaluation details of both the served model and the trained model
                deployed_model_results = read_yaml_file(
                    file_path=self.model_pusher_config.deployed_model_details_path)

                evaluated_model_results = read_yaml_file(
                    file_path=self.model_evaluation_artifact.model_evaluation_result_path
                )

                # Checking to see if the trained model is better than the served model
                if evaluated_model_results["test_accuracy"] > deployed_model_results["test_accuracy"]:
                    # Copy the model and the evaluation details into the desired path
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
            
            logging.info(message)

            return is_pushed, message    
        except Exception as e:
            raise NASException(e, sys) from e


    def initiate_model_push(self) -> ModelPusherArtifact:
        """
        Method Name: initiate_model_push

        Description: This method Combines all the relevant methods to achieve model pushing. 

        return : ModelPusherArtifact: details regarding model pushing
        """
        try:
            is_pushed, message = self.copy_evaluated_model_if_better()

            model_pusher_artifact = ModelPusherArtifact(
                is_pushed=is_pushed,
                message=message
            )
            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise NASException(e, sys) from e
