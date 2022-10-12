
import csv
import sys
import pandas as pd

import cassandra
from cassandra.query import dict_factory
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from NewsArticleSorting.NASException import NASException
from NewsArticleSorting.NASLogger import logging

from NewsArticleSorting.NASEntity.NASConfigEntity import CassandraDatabaseConfig


class NASCassandraDB:

    def __init__(self, 
    cassandra_database_config: CassandraDatabaseConfig) -> None:
        try:
            self.cassandra_database_config = cassandra_database_config
            self.session = self.db_connection()

        except Exception as e:
            raise NASException(e, sys) from e


    def db_connection(self)-> cassandra.cluster.Session:
        try:
            cloud_config = {
                'secure_connect_bundle': self.cassandra_database_config.file_path_secure_connect
            }

            auth_provider = PlainTextAuthProvider(self.cassandra_database_config.cassandra_client_id,
                                                    self.cassandra_database_config.cassandra_client_secret)

            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)

            session = cluster.connect()

            session.row_factory = dict_factory

            session.execute(f"USE {self.cassandra_database_config.keyspace_name}")

            return session

        except Exception as e:
            raise NASException(e, sys) from e

    def create_table(self, column_names: dict):
        try:
            table_creation_query = f"CREATE TABLE IF NOT EXISTS {self.cassandra_database_config.table_name}(id int primary key,"

            for col_name in column_names:
                table_creation_query += f"\"{col_name}\" {column_names[col_name]},"

            table_creation_query = table_creation_query[:-1] + ");"

            print(table_creation_query)

            self.session.execute(table_creation_query)

            self.session.execute(f"truncate table {self.cassandra_database_config.table_name};")

        except Exception as e:
            raise NASException(e, sys) from e

    def insert_valid_data(self, df: pd.DataFrame):
        try:
            
            col_names = "id,"

            for i in list(df.columns):
                col_names += f"\"{str(i).rstrip()}\","

            col_names = col_names[:-1]

            for i in range(len(df)):

                temp_lis = [i+1] + list(df.iloc[i])

                if 'null' in temp_lis:
                    tup = "("
                    for j in temp_lis:
                        if type(j) == str:
                            if j=='null':
                                tup += f"{j},"
                            else:
                                tup += f"'{j}',"
                        else:
                            tup += f"{j},"
                    tup = tup[:-1] + ")" 
                else:
                    tup = tuple(temp_lis)
                insert_query = f"INSERT INTO {self.cassandra_database_config.table_name}({col_names}) VALUES {tup};"  

                self.session.execute(insert_query)  
               
        except Exception as e:
            raise NASException(e, sys) from e
    
    def data_db_to_csv(self, validated_file_path):
        try:
            
            col_name_query = f"select column_name from system_schema.columns where keyspace_name=" \
                                 f"'{self.cassandra_database_config.keyspace_name}' and table_name='{self.cassandra_database_config.table_name}'; "
            
            headers = []
            result = self.session.execute(col_name_query)
            for i in result:

                headers.append(str(i['column_name']))
                print(i['column_name'])
            
            headers.remove('id')

            get_all_data_query = f"select * from {self.cassandra_database_config.table_name};"
            results = self.session.execute(get_all_data_query)

            data = []

            for result in results:
                # List which stores all the information of a row
                row = []
                for header in headers:
                    # Since result is a dictionary with the column names as keys
                    row.append(result[header])
                data.append(row)

            with open(validated_file_path, 'w', newline='') as csv_file:
                # Obtaining the csv writer object to write into relevant csv file
                csv_writer = csv.writer(csv_file)
                # Writing the collumn names
                csv_writer.writerow(headers)
                # Writing the data
                csv_writer.writerows(data)

        except Exception as e:
            raise NASException(e, sys) from e

    def terminate_session(self):
        try:
            self.session.shutdown()

        except Exception as e:
            raise NASException(e, sys) from e