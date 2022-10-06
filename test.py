from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASDataIngestion import NASDataIngestion
from NewsArticleSorting.NASComponents.NASDataValidation import NASDataValidation
from NewsArticleSorting.NASDatabase.NASCassandraDB import NASCassandraDB


if __name__=="__main__":
    nas_config = NASConfiguration()
    data_ingestion_config = nas_config.get_data_ingestion_config()
    data_validation_config = nas_config.get_data_validation_config()

    data_ingestion = NASDataIngestion(data_ingestion_config=data_ingestion_config)

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
    print(data_ingestion_artifact)

    cassandra_db_config = nas_config.get_cassandra_database_config()

    db_operator = NASCassandraDB(cassandra_database_config=cassandra_db_config)

    data_validator = NASDataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                        data_validation_config=data_validation_config,
                                        cassandra_db_operator=db_operator,
                                        use_database=True
                                        )

    data_validation_artifact = data_validator.initiate_validation()

    print(data_validation_artifact)

    