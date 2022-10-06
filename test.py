from NewsArticleSorting.NASConfig.NASConfiguration import NASConfiguration
from NewsArticleSorting.NASComponents.NASDataIngestion import NASDataIngestion


if __name__=="__main__":
    data_ingestion_config = NASConfiguration().get_data_ingestion_config()

    data_ingestion = NASDataIngestion(data_ingestion_config=data_ingestion_config)

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
    print(data_ingestion_artifact)