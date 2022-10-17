import sys

from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASPredictor import NASPredictor
from NewsArticleSorting.NASException import NASException


class NASSingleSentencePredictionPipeline:
    """
    Class Name: NASSingleSentencePredictionPipeline
    Description: Includes the method that is needed predict on a single sentence for News Article Sorting.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """
    def __init__(self) -> None:

        logging.info( f"{'*'*20} Prediction of a Single Sentence log started {'*'*20}")

        self.nas_config = NASConfiguration(is_single_sentence=True,is_training=False)
        self.data_preprocessing_config = self.nas_config.get_data_preprocessing_config()
        self.model_training_config = self.nas_config.get_model_training_config()
        self.model_pusher_config = self.nas_config.get_model_pusher_config()
        self.predictor_config = self.nas_config.get_predictor_config()


    def initiate_single_sentence_prediction_pipeline(self, sentence:str)-> str:
        """
        Method Name: initiate_single_sentence_prediction_pipeline
        Description: This method classifies a single sentence.

        returns: str - the Category of the news article passed.
        """
        try:
            # Prediction
            predictor = NASPredictor(data_preprocessing_config=self.data_preprocessing_config,
                                    model_training_config=self.model_training_config,
                                    model_pusher_config=self.model_pusher_config,
                                    predictor_config=self.predictor_config)

            category = predictor.initiate_prediction_single_sentence(sentence=sentence)

            return category

        except Exception as e:
            raise NASException(e, sys) from e

if __name__ == "__main__":
    pipeline = NASSingleSentencePredictionPipeline()
    sentence = "worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness.  cynthia cooper  worldcom s ex-head of internal accounting  alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (£5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.  prosecution lawyers have argued that mr ebbers orchestrated a series of accounting tricks at worldcom  ordering employees to hide expenses and inflate revenues to meet wall street earnings estimates. but ms cooper  who now runs her own consulting business  told a jury in new york on wednesday that external auditors arthur andersen had approved worldcom s accounting in early 2001 and 2002. she said andersen had given a  green light  to the procedures and practices used by worldcom. mr ebber s lawyers have said he was unaware of the fraud  arguing that auditors did not alert him to any problems.  ms cooper also said that during shareholder meetings mr ebbers often passed over technical questions to the company s finance chief  giving only  brief  answers himself. the prosecution s star witness  former worldcom financial chief scott sullivan  has said that mr ebbers ordered accounting adjustments at the firm  telling him to  hit our books . however  ms cooper said mr sullivan had not mentioned  anything uncomfortable  about worldcom s accounting during a 2001 audit committee meeting. mr ebbers could face a jail sentence of 85 years if convicted of all the charges he is facing. worldcom emerged from bankruptcy protection in 2004  and is now known as mci. last week  mci agreed to a buyout by verizon communications in a deal valued at $6.75bn."
    print(pipeline.initiate_single_sentence_prediction_pipeline(sentence=sentence))