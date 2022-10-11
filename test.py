from NewsArticleSorting.NASComponents.NASModelEvaluation import NASModelEvaluation
from NewsArticleSorting.NASComponents.NASModelPusher import NASModelPusher
from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASDataIngestion import NASDataIngestion
from NewsArticleSorting.NASComponents.NASDataValidation import NASDataValidation
from NewsArticleSorting.NASDatabase.NASCassandraDB import NASCassandraDB
from NewsArticleSorting.NASComponents.NASDataPreprocessing import NASDataPreprocessing
from NewsArticleSorting.NASComponents.NASModelTraining import NASModelTraining


if __name__=="__main__":
    nas_config = NASConfiguration()
    data_ingestion_config = nas_config.get_data_ingestion_config()
    data_validation_config = nas_config.get_data_validation_config()

    data_ingestion = NASDataIngestion(data_ingestion_config=data_ingestion_config)

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
    print(data_ingestion_artifact)

    # cassandra_db_config = nas_config.get_cassandra_database_config()

    # db_operator = NASCassandraDB(cassandra_database_config=cassandra_db_config)

    data_validator = NASDataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                        data_validation_config=data_validation_config,
                                        )

    data_validation_artifact = data_validator.initiate_validation()

    print(data_validation_artifact)


    data_preprocessing_config = nas_config.get_data_preprocessing_config()

    data_preprocessor = NASDataPreprocessing(data_preprocessing_config=data_preprocessing_config,
                                            data_validation_artifact=data_validation_artifact)

    data_preprocessing_artifact = data_preprocessor.initiate_data_preprocessing()

    print(data_preprocessing_artifact)

    model_training_config = nas_config.get_model_training_config()



    model_trainer = NASModelTraining(data_preprocessing_artifact=data_preprocessing_artifact,
                                model_training_config=model_training_config)

    model_trainer_artifact = model_trainer.initiate_model_training()

    print(model_trainer_artifact)

    mode_evaluation_config = nas_config.get_model_evaluation_config()

    model_evaluator = NASModelEvaluation(
        model_training_config=model_training_config,
        model_evaluation_config=mode_evaluation_config,
        data_preprocessing_artifact=data_preprocessing_artifact,
        model_trainer_artifact=model_trainer_artifact
    )

    model_evaluation_artifact = model_evaluator.initiate_evaluation()

    print(model_evaluation_artifact)

    model_pusher_conifg = nas_config.get_model_pusher_config()

    model_pusher = NASModelPusher(
        model_pusher_config=model_pusher_conifg,
        model_trainer_artifact=model_trainer_artifact,
        model_evaluation_artifact=model_evaluation_artifact
    )

    model_pusher_artifact = model_pusher.initiate_model_push()

    print(model_pusher_artifact)

    