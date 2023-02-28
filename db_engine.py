
import urllib.request
import random
import json
import time
import csv
import logging, sys, getopt
import os
import random
import array
import asyncio
import re
import pickle 
from datetime import date, datetime
from multiprocessing import Process
from data_generator import generateUsername, generateServerPassword

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.disable(sys.maxsize)
today = date.today()
todays_date_formatted = today.strftime("%Y-%m-%d")

try:
    from dotenv import load_dotenv
    load_dotenv('./.env')
except Exception as e:
    print("\ninstalling python-dotenv")
    print(os.system("pip install python-dotenv"))
try:
    import asyncpg
except Exception as e:
    print("\ninstalling asyncpg")
    print(os.system("pip install asyncpg"))
try:
    import numpy as np
except Exception as e:
    print("\ninstalling numpy")
    print(os.system("pip install numpy"))
try:
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
except Exception as e:
    print("\ninstalling selenium")
    print(os.system("pip install selenium"))
    print(os.system("pip install webdriver_manager"))
try:
    import pandas as pd
except Exception as e:
    print("\ninstalling pandas")
    print(os.system("pip install pandas"))
try:
    import multiprocessing as mp
except Exception as e:
    print("\ninstalling multiprocessing")
    print(os.system("pip install multiprocessing"))
try:
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
except Exception as e:
    print("\ninstalling plotly")
    print(os.system("pip install plotly"))
    print(os.system("pip install matplotlib"))
try:
    import twint
except Exception as e:
    print("\ninstalling twint")
    os.system("pip install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint")
try:
    import praw
except Exception as e:
    print("\ninstalling praw")
    os.system("pip install praw")
try:
    import faiss
except Exception as e:
    print("\ninstalling faiss")
    print(os.system("pip install faiss-gpu"))

try:
    from transformers import AutoTokenizer, AutoModel, TFAutoModel, pipeline
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification
    from transformers import TFAutoModelForSequenceClassification
except Exception as e:
    print("\ninstalling transformers")
    print(os.system("pip install transformers"))
    print(os.system("pip install sentencepiece"))
    print(os.system("pip install accelerate"))

try:
    from sklearn.svm import SVR
except Exception as e:
    print("\ninstalling scikit-learn")
    print(os.system("pip install scikit-learn"))

try:
    from neuralprophet import NeuralProphet, set_log_level
except Exception as e:
    print("\ninstalling neuralprophet")
    print(os.system("pip install neuralprophet"))

try:
    from scipy.special import softmax
except Exception as e:
    print("\ninstalling scipy")
    print(os.system("pip install scipy"))

try:
    from faker import Faker
except Exception as e:
    print("\ninstalling faker")
    print(os.system("pip install faker"))

try:
    from tinydb import TinyDB, Query 
except Exception as e:
    print("\ninstalling tinydb")
    print(os.system("pip install tinydb"))

try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("\ninstalling pillow")
    print(os.system("pip install --upgrade protobuf==3.20.0"))
    print(os.system("pip install pillow"))

try:
    from wordcloud import WordCloud, STOPWORDS
except Exception as e:
    print("\ninstalling top2vec")
    print(os.system("pip install top2vec"))

try:
    from detoxify import Detoxify
except Exception as e:
    print("\ninstalling detoxify")
    print(os.system("pip install detoxify"))

try:
    import nest_asyncio
except Exception as e:
    print("\ninstalling detoxify")
    print(os.system("pip install nest_asyncio"))

class DB_Engine():

    def __init__(self, user='fuck_your_face', password='fuck_your_face', database='fuck_your_face', host='127.0.0.1'):
        self.user = user
        self.password = password
        self.database = database
        self.host = host
        
    async def checkEnv(self, createIfFail=True):
        try:
            db_username = str(os.getenv('DB_USERNAME')).lower()
            db_pass = os.getenv('DB_PASSWORD')
            db_database = os.getenv('DB_DATABASE')
            db_server = os.getenv('DB_HOST')
            if(db_username is not None and db_pass is not None and db_database is not None and db_server is not None):
                response_obj = {
                    "db_username":db_username,
                    "db_pass": db_pass,
                    "db_database": db_database,
                    "db_server": db_server,
                }
                await self.initializeConnection()
                return response_obj
            else:
                raise Exception("No Credentials to pull.. Check env file")
        except Exception as e:
            if(createIfFail):
                response_obj = {}
                response_obj["db_pass"] = generateServerPassword()
                response_obj["db_username"] = str(generateUsername()).lower()
                response_obj["db_database"] = str(generateUsername()).lower() #"sociality_test"
                response_obj["db_server"] = "127.0.0.1"    
                print("\n=========================================== SAVING POSTGRES CONFIGURATION ===========================================")
                print("\nUsername: : {}".format(response_obj["db_username"]))
                print("\nPassword: : {}".format(response_obj["db_pass"]))
                print("\nDatabase: : {}".format(response_obj["db_database"]))
                print("\nHost: : {}".format(response_obj["db_server"]))
                print("\n=======================================================================================================================")
                f = open("./.env", "w")
                env_file_string = """DB_USERNAME={0}\nDB_PASSWORD={1}\nDB_HOST={2}\nDB_DATABASE={3}""".format(response_obj["db_username"], response_obj["db_pass"], response_obj["db_server"], response_obj["db_database"] )
                f.write(env_file_string)
                f.close()
                print("\n=================================")
                print("\nInstalling Postgres...")
                print("\n=================================")
                self.user = response_obj["db_username"]
                self.password = response_obj["db_pass"]
                self.database = response_obj["db_database"]
                self.host = response_obj["db_server"]
                await self.createPostgresServer()

                print("\n=================================")
                print("\nSetting up tables...")
                print("\n=================================")
                
                await self.initializeConnection()
                print("\n=================================")
                print("\nInstalling PM2...")
                print("\n=================================")
                try:
                    await Task_Scheduler().install()
                except Exception as e:
                    print("\nException installing pm2")
                    print(e)
                if(response_obj["db_username"] is not None and response_obj["db_pass"] is not None and response_obj["db_database"] is not None and response_obj["db_server"] is not None):
                    response_obj = {
                        "db_username":str(response_obj["db_username"]).lower(),
                        "db_pass": response_obj["db_pass"],
                        "db_database": str(response_obj["db_database"]).lower(),
                        "db_server": response_obj["db_server"],
                    }
                    return response_obj
            else:
                raise Exception(e)
                print(e)

    async def createPostgresServer(self):
        initial_db_name = self.database 
        db_admin_user = self.user
        db_password = self.password
        command1 = """
        # Installing Postgres
        sudo apt install postgresql postgresql-contrib -y
        sudo systemctl start postgresql.service
         """
        command1_output = os.system(command1)
        command2 =  """echo "SELECT 'CREATE DATABASE """ + initial_db_name + """' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '""" + initial_db_name + """')\\gexec" | sudo -i -u postgres psql"""
        print(command2)
        command2_output = os.system(command2)
        command3 =  """echo "CREATE USER """ + db_admin_user + """ WITH password '""" + db_password + """'\\gexec" | sudo -i -u postgres psql"""
        print(command3)
        command3_output = os.system(command3)
        command4 =  'sudo -i -u postgres psql -c "GRANT ALL privileges ON DATABASE ' + initial_db_name + " TO " + db_admin_user + '";'
        print(command4)
        command4_output = os.system(command4)

        return command4_output

    async def backupPostgresServer(self):
        db_name = self.database 
        db_admin_user = self.user
        db_password = self.password
        db_host = self.host
        command1 =   "echo '" + db_host + ":5432:"+ db_name + ":" +db_admin_user + ":" + db_password + "'  >> ~/.pgpass && chmod 600 ~/.pgpass"
        command1_output = os.system(command1)
        command2 =  """pg_dump -f ./postgres_backup-tmp.tar -C -F t -d """ + db_name + " -h " + db_host + " -U " + db_admin_user 
        command2_output = os.system(command2)
        return command2_output

    async def pullConnection(self):
        def _encoder(value):
            return b'\x01' + json.dumps(value).encode('utf-8')
        def _decoder(value):
            return json.loads(value[1:].decode('utf-8'))
        conn = await asyncpg.connect(user=self.user, password=self.password, database=self.database, host=self.host)
        await conn.set_type_codec('jsonb', encoder=_encoder, decoder=_decoder, schema='pg_catalog', format='binary')
        return conn  

    async def initializeConnection(self):
        def _encoder(value):
            return b'\x01' + json.dumps(value).encode('utf-8')

        def _decoder(value):
            return json.loads(value[1:].decode('utf-8'))
        try:
            conn = await asyncpg.connect(user=self.user, password=self.password,
                                database=self.database, host=self.host)
            await conn.set_type_codec('jsonb', encoder=_encoder, decoder=_decoder, schema='pg_catalog', format='binary')
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS sociality(
                        id SERIAL PRIMARY KEY,
                        messageId TEXT NOT NULL,
                        username TEXT NOT NULL,
                        text TEXT NOT NULL,
                        date_posted TEXT NOT NULL,
                        source TEXT NOT NULL,
                        dataType TEXT NOT NULL,
                        data JSONB NOT NULL,
                        model_results JSONB,
                        entities_extracted JSONB,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
            except Exception as e:
                print("\nerror creating table ")
                print(e)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results(
                        id SERIAL PRIMARY KEY,
                        messageId TEXT NOT NULL,
                        username TEXT NOT NULL,
                        text TEXT NOT NULL,
                        date_posted TEXT NOT NULL,
                        source TEXT NOT NULL,
                        dataType TEXT NOT NULL,
                        data JSONB NOT NULL,
                        model_results JSONB,
                        entities_extracted JSONB,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
            except Exception as e:
                print("\nerror creating table ")
                print(e)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS similar_users(
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        similar_Profiles JSONB NOT NULL,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
            except Exception as e:
                print("\nerror creating table ")
                print(e)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS metric_user_similarities(
                        id SERIAL PRIMARY KEY,
                        model TEXT NOT NULL,
                        labels JSONB NOT NULL,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
            except Exception as e:
                print("\nerror creating table ")
                print(e)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_predictions(
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        time_frame TEXT NOT NULL,
                        selected_periods TEXT NOT NULL,
                        username TEXT NOT NULL,
                        metric_predictions JSONB NOT NULL,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
            except Exception as e:
                print("\nerror creating table ")
                print(e)
            return conn
        except Exception as e:
                    print("\nerror creating table ")
                    print(e)

    async def wipe_db(self):
        db_username = os.getenv('DB_USERNAME')
        db_pass = os.getenv('DB_PASSWORD')
        db_database = os.getenv('DB_DATABASE')
        db_server = os.getenv('DB_HOST')
        try:
            print("\nDropping database {0}".format(db_database))
            try:
                print(os.system("sudo -iu postgres psql -c \"DROP DATABASE IF EXISTS {0}\"".format(db_database)))
            except Exception as e:
                print("\nError dropping database {0}".format(db_database))
                print(e)
                
            print("\nDropping db_username {0}".format(db_username))

            try:
                print(os.system("sudo -iu postgres psql -c \"DROP USER IF EXISTS {0}\"".format(db_username)))
            except Exception as e:
                print("\nError dropping USER {0}".format(db_username))
                print(e)
            print("\nrevoking priviledges for {0}".format(db_username))

            try:
                print(os.system('sudo -i -u postgres psql -c "REVOKE ALL privileges ON DATABASE ' + db_database + " FROM " + db_username + ';"'))
            except Exception as e:
                print("\nError revoking priviledges for user {0}".format(db_username))
                print(e)
            try:
                f = open("./.env", "w")
                f.write("")
                f.close()
            except Exception as e:
                print("\nError clearing file")
                print(e)
        except Exception as e:
                print("\nError removing database... You check the env file?")
                print(e)

    async def search_for_record(self, table_name='sociality', username = False, message_id = False):
        conn = await self.pullConnection() 
        if(username):
            query = 'SELECT * FROM ' + table_name + " WHERE username = '" + username + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        elif(message_id):
            query = 'SELECT * FROM ' + table_name + " WHERE messageid = '" + message_id + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        else:
            query = 'SELECT * FROM ' + table_name 
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result

    async def run_custom_query(self, query):
        try:
            conn = await self.pullConnection() 
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        except Exception as e:
            print("\nError running custom query")
            print(e)
            return e
    async def pull_analytics(self, query_type='metric_user_similarities', username = False, message_id = False):
        conn = await self.pullConnection() 
        if(query_type == "metric_user_similarities"):
            query = 'SELECT * FROM metric_user_similarities'
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        elif(query_type == "model_predictions"):
            query = 'SELECT * FROM model_predictions'
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        elif(query_type == "similar_users"):
            query = 'SELECT * FROM similar_users'
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        else:
            query = 'SELECT * FROM similar_users'
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
            
    async def pull_analytics_user(self, query_type='metric_user_similarities', username = False, message_id = False):
        conn = await self.pullConnection() 
        if(query_type == "metric_user_similarities"):
            query = "SELECT * FROM metric_user_similarities where username = '" + username + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        elif(query_type == "model_predictions"):
            query = "SELECT * FROM model_predictions where username = '" + username + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        elif(query_type == "similar_users"):
            query = "SELECT * FROM similar_users where username = '" + username + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result
        else:
            query = "SELECT * FROM similar_users where username = '" + username + "'"
            query_result = await conn.fetch(query)
            await conn.close()
            return query_result

    async def insert_similar_user_records(self, similar_users, dropTable=False):
        conn = await self.pullConnection() 
        if(dropTable):
            await conn.execute("DROP TABLE similar_users")
            await conn.close()
        else:        
            for user_data in similar_users:
                username = user_data["user_searched"]
                similar_profiles = user_data["similar_Profiles"]
                a_last_updated = datetime.utcnow()
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS similar_users(
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        similar_Profiles JSONB NOT NULL,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                        )
                ''')
                await conn.execute('''
                            INSERT INTO similar_users(username, similar_profiles, last_updated) VALUES($1, $2, $3)
                        ''', username, similar_profiles, a_last_updated)
            await conn.close()

    async def insert_similar_metric_records(self, similar_metrics, dropTable=False):
        conn = await self.pullConnection() 
        if(dropTable):
            await conn.execute("DROP TABLE metric_user_similarities")
            await conn.close()
        else:
            for user_data in similar_metrics:
                model = user_data["model"]
                labels = user_data["labels"]
                a_last_updated = datetime.utcnow()
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS metric_user_similarities(
                        id SERIAL PRIMARY KEY,
                        model TEXT NOT NULL,
                        labels JSONB NOT NULL,
                        last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                    )
                ''')
                await conn.execute('''
                        INSERT INTO metric_user_similarities(model, labels, last_updated) VALUES($1, $2, $3)
                    ''', model, labels, a_last_updated)
            await conn.close()

    async def insert_model_prediction_record(self, model_prediction, dropTable=False):
        conn = await self.pullConnection() 
        if(dropTable):
            await conn.execute("DROP TABLE model_predictions")
            await conn.close()
        else:
            model = model_prediction["model"]
            metric = model_prediction["metric"]
            time_frame = model_prediction["time_frame"]
            selected_periods = model_prediction["selected_periods"]
            username = model_prediction["username"]
            metric_prediction = model_prediction["metric_prediction"]
            a_last_updated = datetime.utcnow()
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions(
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    time_frame TEXT NOT NULL,
                    selected_periods TEXT NOT NULL,
                    username TEXT NOT NULL,
                    metric_predictions JSONB NOT NULL,
                    last_updated TIMESTAMP(0) NOT NULL DEFAULT NOW()
                )
            ''')
            await conn.execute('''
                        INSERT INTO model_predictions(model_name, metric_name, time_frame, selected_periods, username, metric_predictions, last_updated) VALUES($1, $2, $3, $4, $5, $6, $7)
                    ''', model, metric, time_frame, str(selected_periods), username, metric_prediction, a_last_updated)
            await conn.close()

    async def insert_social_record(self, record_to_insert, table_name='sociality'):
        conn = await self.pullConnection() 
        message_id = record_to_insert["messageId"]
        username = record_to_insert["username"]
        text = record_to_insert["text"]
        data = record_to_insert["data"]
        source = record_to_insert["source"]
        dataType = record_to_insert["dataType"]
        model_results = None
        entities_extracted = None
        if(table_name != "sociality"):
            model_results = record_to_insert["model_results"]
            entities_extracted = record_to_insert["entities_extracted"]
        else:
            model_results = []
            entities_extracted = []
        date_posted = record_to_insert["dateTime"]
        a_last_updated = datetime.utcnow()
        conn = await self.pullConnection() 
        query_result = await self.search_for_record(table_name, False, message_id)
        if(not query_result):
            print('Inserting ' + username + " - " + text)
            if(table_name == "sociality"):
                await conn.execute('''
                    INSERT INTO sociality(messageId, username, text, date_posted, source, dataType, data, model_results, entities_extracted, last_updated) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''', message_id, username, text, date_posted, source, dataType, data, model_results,entities_extracted, a_last_updated)
                await conn.close()
            elif(table_name == "analysis_results"):
                await conn.execute('''
                    INSERT INTO analysis_results(messageId, username, text, date_posted, source, dataType, data, model_results, entities_extracted, last_updated) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''', message_id, username, text, date_posted, source, dataType, data, model_results,entities_extracted, a_last_updated)
                await conn.close()

    async def syncFileWithPostgres(self, main_file='./tinydb.json', table_name="analysis_results"):
        tinydb = TinyDB(main_file)
        tinydb_table = tinydb.table(table_name)
        for row in tinydb_table.all()[::-1]:
            messageId = row["messageId"]
            conn = await self.pullConnection() 
            query_result =  await self.search_for_record(table_name, False, messageId)
            if(not query_result):
                await self.insert_social_record(row, table_name)
         
            await conn.close()
    async def syncPostgresToFile(self, main_file='./tinydb.json', table_name="analysis_results"):
        userData = await self.search_for_record('analysis_results')
        tinydb = TinyDB(main_file)
        tinydb_table = tinydb.table(table_name)
    
        for record in userData:
            message_id = record["messageid"]
            recordToSearch = Query()
            insertedRecord = tinydb_table.search((recordToSearch.messageId == message_id))
            if(not insertedRecord):
                username = record["username"]
                text = record["text"]
                data = record["data"]
                source = record["source"]
                dataType = record["datatype"]
                model_results = record["model_results"]
                dateTime = record["date_posted"]
                entities_extracted = record["entities_extracted"]
                record_To_Insert = {
                    "username": username,
                    "messageId": message_id,
                    "text" : text,
                    "source": source,
                    "dataType": dataType,
                    "data": data,
                    "model_results": model_results,
                    "entities_extracted": entities_extracted,
                    "dateTime": dateTime,
                }
                tinydb_table.insert(record_To_Insert)
                
    async def restoreDatabaseFromBackup(self, backup_file_path):
        command_string = "sudo -i -u postgres psql {} < {}".format(self.database,backup_file_path)
        command_output = os.system(command_string)
        print(command_output)
   