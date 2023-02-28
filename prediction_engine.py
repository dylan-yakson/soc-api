import urllib.request
import random
import json
import time
import csv
import logging, sys, getopt
import os
import random
import array
from collections import Counter
from datetime import date, datetime
from multiprocessing import Process, Lock

from nlp_models import pullEntitiesFromText
import asyncio
import re
import pickle 

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.disable(sys.maxsize)
today = date.today()
todays_date_formatted = today.strftime("%Y-%m-%d")
# ===============================================================================================================
#                                   Check & Install Dependencies
# ===============================================================================================================
#
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
    import pandas as pd
except Exception as e:
    print("\ninstalling pandas")
    print(os.system("pip install pandas"))
try:
    import multiprocessing as mp
    from multiprocessing import Process, Lock
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

nest_asyncio.apply()

    
class Prediction_Engine():
    def __init__(self, modelFunctionList, db_engine):
        self.modelFunctionList = modelFunctionList
        self.db_engine = db_engine
    #================================            
    # Data Retrieval            
    #================================            
    async def pullModelData(self, username, table="sociality"):
        userData = await self.db_engine.search_for_record(table, username)
        return userData
                      
    async def pullListOfUsers(self, tableName):
        Users_Result_Array = []
        userData = await self.db_engine.search_for_record(tableName)
        if(userData):
            for record in userData:
                locatedUser = False
                for user in Users_Result_Array:
                    if(user == record["username"]):
                        locatedUser = True
                if(not locatedUser):
                    Users_Result_Array.append(record["username"])
        return Users_Result_Array
    
    async def pullListOfModels(self):
        return self.modelFunctionList

    async def extractEntitiesForUser(self,username):
        userData = await self.db_engine.search_for_record('analysis_results', username)
        Formatted_Entity_Info = []
        for record in userData:
            try:
                entity_data = record["entities_extracted"]
                for entity in entity_data:
                    Formatted_Entity_Info.append(entity)
            except Exception as e:
                print("\nException pulling entity data or none existant for text")
                print(record)
                print(e)
        Formatted_Entity_Info.sort(key=lambda x: x["score"], reverse=True)        
        return Formatted_Entity_Info

    #================================            
    # Data Format            
    #================================            
    async def formatUserSocialDataForInternalPostComparison(self, userData):
            Formatted_Result_List_After_Analysis = False
            Formatted_Model_Index_List_After_Analysis = False
            Formatted_Text_Index_List_After_Analysis = False

            for record in userData:
                    model_results = record["model_results"]
                    record_text = record["text"]
                    user_name = record["username"]
                    dateTime = record["date_posted"]
                    modelNum = len(model_results)
                    if(Formatted_Result_List_After_Analysis == False):
                        Formatted_Result_List_After_Analysis = [ [] for _ in range(modelNum) ]
                        Formatted_Model_Index_List_After_Analysis = [ [] for _ in range(modelNum) ]
                        Formatted_Text_Index_List_After_Analysis = [ [] for _ in range(modelNum) ]

                    staging_metric_list = []
                    for iterator1 in range(0, len(model_results)):
                        row = model_results[iterator1]
                        analysis = row["results"]
                        if(len(analysis) == 1):
                            analysis = analysis[0]
                            if(len(analysis) == 1):
                                analysis = analysis[0]
                        for iterator2 in range(0, len(analysis)):
                            scoreObject = analysis[iterator2]
                            try:
                                label = scoreObject["label"]
                                value = scoreObject["score"]
                                staging_metric_list.append(value) 
                            except Exception as e:
                                print("\nAn exception occurred when formatting social data")
                                print("\nanalysis Object:")
                                print(analysis)
                                print(e) 
                        Formatted_Result_List_After_Analysis[iterator1].append(staging_metric_list)
                        Formatted_Text_Index_List_After_Analysis[iterator1].append(record_text)
                        Formatted_Model_Index_List_After_Analysis[iterator1] = row["modelName"]
            response_obj = {
                "model_index": Formatted_Model_Index_List_After_Analysis,
                "model_results": Formatted_Result_List_After_Analysis,
                "record_index": Formatted_Text_Index_List_After_Analysis
            }
            return response_obj

    async def formatUserSocialData(self, userData, modelNumber):
        Formatted_Result_List_After_Analysis = []
        for iterator1 in range(1, len(userData)):
            row = userData[iterator1]
            modelResults = row["model_results"]
            for modelResult in modelResults:
                if(modelResult["modelId"] == modelNumber):
                    analysis = modelResult["results"]
                    dateTimeValue = row["dateTime"]
                    dateTimeValue = datetime.strptime(dateTimeValue, '%Y-%m-%d %H:%M:%S')
                    FormattedModelResultArray=[]
                    for iterator2 in range(0, len(analysis)):
                        scoreObject = analysis[iterator2]
                        try:
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            Formatted_Result_List_After_Analysis.append([iterator1,iterator2,value]) 
                        except Exception as e:
                            print("\nAn exception occurred when formatting social data")
                            print("\nanalysis Object:")
                            print(analysis)
                            print(e)
        return Formatted_Result_List_After_Analysis
        
    async def formatUserSocialDataImageGeneration(self, userData):
        Formatted_Result_List_After_Analysis = False
        for record in userData:
                model_results = record["model_results"]
                modelNum = len(model_results)
                if(Formatted_Result_List_After_Analysis == False):
                    Formatted_Result_List_After_Analysis = [ [] for _ in range(modelNum) ]
                for iterator1 in range(0, len(model_results)):
                    row = model_results[iterator1]
                    analysis = row["results"]
                    model_name = row["modelName"]
                    if(len(analysis) == 1):
                            analysis = analysis[0]
                            if(len(analysis) == 1):
                                analysis = analysis[0]
                    staging_metric_list = []
                    for iterator2 in range(0, len(analysis)):
                        scoreObject = analysis[iterator2]
                        try:
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            staging_metric_list.append(value) 
                        except Exception as e:
                            print("\nAn exception occurred when formatting social data for image generation")
                            print("\nanalysis Object:")
                            print(analysis)
                            print(e) 
                    Formatted_Result_List_After_Analysis[iterator1].append({"Model": model_name, "results": staging_metric_list})
        return Formatted_Result_List_After_Analysis
    #================================            
    # NLP & Models         
    #================================ 
    async def runModel(self, textToProcess):
        ResultList = []
        async def modelFunction(text, l,result_list):
            model_results = await modelFunctionItem.RunModel(textToProcess)
            modelFunctionInfo = await modelFunctionItem.modelInfo()
            formatted_model_response_object = {
                "modelName": modelFunctionInfo["name"],
                "modelId": modelFunctionInfo["id"],
                "results": model_results
            }
            l.acquire()
            ResultList.append(formatted_model_response_object)
            l.release()
        if(len(textToProcess) >= 581):
                    textToProcess = textToProcess[ 0 : 580 ]
        for modelFunctionItem in self.modelFunctionList:
            lock = Lock()
            try:
                p = Process(target = asyncio.run(modelFunction(textToProcess,lock, ResultList)))
                p.start()

            except Exception as e:
                print(e)
        p.join()
        # print("RESULTS LIST", ResultList)
        return ResultList

    async def runModelSingleUserFunction(self, modelData, username, entityExtraction=True):
        text = modelData["text"]
        messageId = modelData["messageid"]
        print("\n\n",username,": ",messageId,"\n", text,"\n\n")
        record_To_Insert = modelData
        insertedRecord = await self.db_engine.search_for_record('analysis_results', False, messageId)
        if(not insertedRecord):
            model_results = await self.runModel(text)
            record_To_Insert = {
                "messageId": messageId,
                "text": text,
                "username": modelData["username"],
                "data": modelData["data"],
                "model_results": model_results,
                "source": modelData["source"],
                "dataType": modelData["datatype"],
                "dateTime": modelData["date_posted"],
                "entities_extracted": modelData["entities_extracted"],
            }
            if( entityExtraction):
                try:
                    entities_extracted = await pullEntitiesFromText(text)
                    record_To_Insert["entities_extracted"] = entities_extracted
                except Exception as e:
                    print("\n=================================")
                    print("\nError runModelsOnSingleScrapedUser")
                    print(e)
                    print("\n=================================")
            await self.db_engine.insert_social_record(record_To_Insert, 'analysis_results')

    async def runModelsOnSingleScrapedUser(self, username, entityExtraction=True):
        formatted_model_data = await self.db_engine.search_for_record('sociality', username)
        for modelData in formatted_model_data:
            t1 = Process(target=asyncio.run(self.runModelSingleUserFunction(modelData, username, entityExtraction)))
            t1.start()
        for modelData in formatted_model_data[::-1]:
            t1 = Process(target=asyncio.run(self.runModelSingleUserFunction(modelData, username, entityExtraction)))
            t1.start()
                
    #================================            
    # Imagee & Data Export          
    #================================ 
    async def concat_images(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
    
    async def genColors(self, values):
        colors=[]
        for i in range(0,len(values)):
            r =random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            rgb = (r,g,b)
            colors.append(rgb)
        return colors
    
    async def generateImage(self, values, colors, width, height):
        img  = Image.new( mode = "RGB", size = (width, height) )
        draw = ImageDraw.Draw(img)
        numFeatures = len(values)
        for i in range(0,numFeatures):
            x1=(width/numFeatures) * i
            x2=((width/numFeatures) * i) + ((width/numFeatures) * values[i])
            draw.rectangle((x1, 0, x2, height), fill=colors[i], outline=(255, 255, 255))
        return img
    
    async def createImageFromSocialModelData(self, username):
        userData = await self.db_engine.search_for_record('analysis_results', username)
        formatted_model_data = await self.formatUserSocialDataImageGeneration(userData)
        final_picture_array = []
        for modelData in formatted_model_data:
            try:
                width = 60
                height = 10
                finalImage = Image.new( mode = "RGB", size = (width, height) )
                colors = False
                model_name = False
                for modelResult in modelData:
                    if(colors == False):
                        colors = await self.genColors(modelResult["results"])
                    if(model_name == False):
                        model_name = modelResult["Model"]
                    numFeatures = len(modelResult["results"])
                    img = await self.generateImage(modelResult["results"],colors, width, height)
                    finalImage = await self.concat_images(img,finalImage)
                final_picture_array.append({"image": finalImage, "model": model_name})
            except Exception as e:
                print("\nexception in createImageFromSocialModelData")
                print(e)
        return final_picture_array
        
    async def create_social_graph(self, username):
        cnt1 = Counter()
        recipients = []
        userData = await self.db_engine.search_for_record('analysis_results')
        for record in userData:
            if(record["datatype"] != "submission"):
                print(record)
                full_record = record["data"]
                print(full_record)
                recipient = full_record
        
    async def createGIFFromSocialModelData(self, username):
        userData = await self.db_engine.search_for_record('analysis_results', username)
        formatted_model_data = await self.formatUserSocialDataImageGeneration(userData)
        base_path = os.path.expanduser(os.path.join('./results'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        user_path = os.path.expanduser(os.path.join(base_path,username))
        user_path_exist = os.path.exists(base_path)
        if(not user_path_exist):
            os.makedirs(user_path) 
        final_picture_array = []
        for modelData in formatted_model_data:
            try:
                width = 60
                height = 10
                finalImages = []
                colors = False
                model_name = False
                for modelResult in modelData:
                    if(colors == False):
                        colors = await self.genColors(modelResult["results"])
                    if(model_name == False):
                        model_name = modelResult["Model"]
                    numFeatures = len(modelResult["results"])
                    img = await self.generateImage(modelResult["results"],colors, width, height)
                    finalImages.append(img)
                im = Image.new( mode = "RGB", size = (900, 1400) )
                gif_filename = user_path + "/" + username + "_" + model_name +"_progression.gif" 
                im.save(gif_filename, save_all=True, append_images=finalImages)
                final_picture_array.append(im)
            except Exception as e:
                print("\nexception in createImageFromSocialModelData")
                print(e)
        return final_picture_array

    #================================            
    # Analytics         
    #================================ 
    async def findTopPostsPerModelMetric(self, username=False, number_similar_profiles=4, desired_metric_threshold=.5):
            userData = False
            if(username):
                userData = await self.db_engine.search_for_record('analysis_results', username)
            else:
                userData = await self.db_engine.search_for_record('analysis_results')
                
            Formatted_User_Data_Array = {}
            Formatted_User_Values_Array = {}
            for record in userData:
                    model_results = record["model_results"]
                    record_text = record["text"]
                    user_name = record["username"]
                    dateTime = record["date_posted"]
                    modelNum = len(model_results)
                    for iterator1 in range(0, len(model_results)):
                        row = model_results[iterator1]
                        analysis = row["results"]
                        model_name = row["modelName"]
                        staging_metric_list = []
                        staging_metric_labels = []
                        staging_metric_posts = []

                        staging_metric_list.append(dateTime)
                        staging_metric_labels.append('dateTime')
                        
                        if(len(analysis) == 1):
                            analysis = analysis[0]
                        for iterator2 in range(0, len(analysis)):
                            scoreObject = analysis[iterator2]
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            # if(value > desired_metric_threshold):
                            if(True):
                                staging_metric_list.append(value) 
                                staging_metric_labels.append(label)
                                staging_metric_posts.append(record_text)
                                
                                if(not user_name in Formatted_User_Data_Array):
                                    Formatted_User_Data_Array[user_name] = {}
                                    Formatted_User_Data_Array[user_name][model_name] = {}
                                    Formatted_User_Data_Array[user_name][model_name][label] = []
                                    
                                    Formatted_User_Values_Array[user_name] = {}
                                    Formatted_User_Values_Array[user_name][model_name] = {}
                                    Formatted_User_Values_Array[user_name][model_name][label] = []
                                else:
                                    if(not model_name in Formatted_User_Data_Array[user_name]):
                                        Formatted_User_Data_Array[user_name][model_name] = {}
                                        Formatted_User_Data_Array[user_name][model_name][label] = []
                                        Formatted_User_Data_Array[user_name][model_name][label].append({"text": record_text, "date" :dateTime, "value": value})

                                        Formatted_User_Values_Array[user_name][model_name] = {}
                                        Formatted_User_Values_Array[user_name][model_name][label] = []
                                        Formatted_User_Data_Array[user_name][model_name][label].append(value)
                                    else:
                                        if(not label in Formatted_User_Data_Array[user_name][model_name]):
                                            Formatted_User_Data_Array[user_name][model_name][label] = []
                                            Formatted_User_Data_Array[user_name][model_name][label].append({"text": record_text, "date" :dateTime, "value": value})
                                            Formatted_User_Values_Array[user_name][model_name][label] = []
                                            Formatted_User_Data_Array[user_name][model_name][label].append(value)
                                        else:
                                            Formatted_User_Data_Array[user_name][model_name][label].append({"text": record_text, "date" :dateTime, "value": value})
                                            Formatted_User_Data_Array[user_name][model_name][label].append(value)
            response_obj = {
                    "Model_Results": Formatted_User_Data_Array,
                    "Model_Results_Index": Formatted_User_Values_Array,
            }
            return response_obj

    async def findSimilarPostsOfUser(self, username, number_similar_profiles=4):
        userData = await self.db_engine.search_for_record('analysis_results', username)
        formatted_model_data = await self.formatUserSocialDataForInternalPostComparison(userData) 
        k = number_similar_profiles
        Similar_Results_Per_Model_Array = []
        model_index = formatted_model_data["model_index"]
        model_result_data = formatted_model_data["model_results"]

        for iterator1 in range(0, len(model_result_data)):
            scoreObject = analysis[iterator1]
            model_record_index_data = formatted_model_data["record_index"][modelIterator]
            model_name = model_index[modelIterator]
            Formatted_Model_Data = np.array(modelData, dtype="float32")
            numFeatures = len(Formatted_Model_Data[0])
            index = faiss.IndexFlatL2(numFeatures)
            index.add(Formatted_Model_Data)
            D, I = index.search(Formatted_Model_Data, k)
            model_results = []
            for SimilarRecordIndex in I:
                recordIndex = SimilarRecordIndex[0]
                selectedRecord = model_record_index_data[recordIndex]
                model_results.append(selectedRecord)
            response_obj = {
                "model_name": model_name,
                "similar_posts_index": I,
                "similar_posts": model_results,
            }
            formatted_similar_posts = []
            for post in response_obj["similar_posts_index"]:
                post_index = response_obj["similar_posts"]
                similar_posts_tmp = []
                for post_record in post:
                    similar_posts_tmp.append(response_obj["similar_posts"][post_record])
                formatted_similar_posts.append(similar_posts_tmp)
            response_obj["similar_posts"] = formatted_similar_posts
            Similar_Results_Per_Model_Array.append(response_obj)
        return Similar_Results_Per_Model_Array


    async def generateWordCloudForUser(self,username, skip_word_array=['http','https']):
        TextArray = []
        userData = await self.db_engine.search_for_record('analysis_results')

        base_path = os.path.join("./results")
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            await os.makedirs(base_path)
        user_path = os.path.join(base_path,username)
        user_path_exist = os.path.exists(base_path)
        if(not user_path_exist):
            await os.makedirs(user_path)
        for data in userData:
            TextArray.append(data["text"].lower())
        comment_words = ''
        comment_words += " ".join(TextArray)+" "
        stopwords = set(STOPWORDS)
        for word in skip_word_array:
            stopwords.add(word)
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 10,
                        min_word_length=4).generate(comment_words)
        # plot the WordCloud image                      
        plt.figure(figsize = (8, 8), facecolor = None)
        # plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        # plt.show()
        filename1 = user_path + "/" + username + "_WORD_CLOUD.png" 
        plt.savefig(filename1)
        return plt

    #================================            
    # Simularities         
    #================================         
    async def formatUserSocialDataForIdentityModelComparison(self, userData, desired_time_frame_frequency='30min', desired_selected_periods=100):
        Formatted_User_Data_Array = {}
        Formatted_User_Values_Array = {}
        DF_Formatted_User_Values_Array = {}
        for record in userData:
            model_results = record["model_results"]
            record_text = record["text"]
            user_name = record["username"]
            dateTime = record["date_posted"]
            if(model_results):
                try:
                    for iterator1 in range(0, len(model_results)):
                        row = model_results[iterator1]
                        analysis = row["results"]
                        model_name = row["modelName"]
                        if(len(analysis) == 1):
                            analysis = analysis[0]
                        for iterator2 in range(0, len(analysis)):
                            scoreObject = analysis[iterator2]
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            if(not model_name in Formatted_User_Data_Array):
                                Formatted_User_Data_Array[model_name] = {}
                                Formatted_User_Data_Array[model_name][label] = {}
                                Formatted_User_Values_Array[model_name] = {}
                                Formatted_User_Values_Array[model_name][label] = {}
                                Formatted_User_Values_Array[model_name][label][user_name] = []
                                Formatted_User_Values_Array[model_name][label][user_name].append(value)
                            else:
                                if (not label in Formatted_User_Data_Array[model_name]):
                                    Formatted_User_Data_Array[model_name][label] = {}
                                    Formatted_User_Values_Array[model_name][label] = {}
                                    
                                if(not user_name in Formatted_User_Data_Array[model_name][label]):
                                    Formatted_User_Data_Array[model_name][label][user_name] = []
                                    Formatted_User_Data_Array[model_name][label][user_name].append({"text": record_text, "date" :dateTime, "value": value})
                                    Formatted_User_Values_Array[model_name][label][user_name] = []
                                    Formatted_User_Values_Array[model_name][label][user_name].append(value)
                                else:
                                    Formatted_User_Data_Array[model_name][label][user_name].append({"text": record_text, "date" :dateTime, "value": value})
                                    Formatted_User_Values_Array[model_name][label][user_name].append(value)
                except Exception as e:
                    print("Exception in formatUserSocialDataForIdentityModelComparison")

            # Find simiar users that match certain model metric
        final_response_object = []
        for model in Formatted_User_Values_Array:
            model_results = Formatted_User_Data_Array[model]
            formatted_model_array = []
            for label in model_results:
                Formatted_User_Averages_Array = []
                Formatted_User_Index_Array = []
                Formatted_User_Values_Array = []
                user_label_data = model_results[label]
                for username in user_label_data:
                    user_label_result_array = user_label_data[username]
                    tmp_array = []
                    for record in user_label_result_array:
                        tmp_array.append([record["date"],record["value"]])

                    Identity_Df = np.array(tmp_array)
                    df = pd.DataFrame(Identity_Df, columns = ["date","value"])
                    df.drop_duplicates(inplace=True)
                    df['date'] = pd.to_datetime(df['date'])
                    df['date'] = pd.DatetimeIndex(df['date'])
                    df.set_index('date').resample(desired_time_frame_frequency).mean().ffill().bfill()
                    # Only grab last x records
                    df.iloc[0:desired_selected_periods]
                    del df[df.columns[0]]
                    df = df.astype(float)
                    model_results_resampled = df.to_numpy()
                    average_of_averages_array = np.mean(model_results_resampled, axis=0)
                    Formatted_User_Averages_Array.append(average_of_averages_array)
                    Formatted_User_Values_Array.append(tmp_array)
                    Formatted_User_Index_Array.append(username)

                formatted_model_array.append({"label": label, "values": Formatted_User_Values_Array, "averages": Formatted_User_Averages_Array, "index": Formatted_User_Index_Array})
            final_response_object.append({"model": model, "results": formatted_model_array})
        return final_response_object
    
    async def findSimilarUsersPerModelMetric(self, number_similar_profiles=4, desired_time_frame_frequency="30T", desired_selected_periods=30, desired_metric_threshold=.5):
            userData = await self.db_engine.search_for_record('analysis_results')
            user_metric_data = await self.formatUserSocialDataForIdentityModelComparison(userData,desired_time_frame_frequency,desired_selected_periods)
            Final_Response_Object = []
            await self.db_engine.insert_similar_metric_records(False, True)
            for model_averages_data in user_metric_data:
                model_name = model_averages_data["model"]
                model_label_results = model_averages_data["results"]
                Model_Response_Object = []
                for label_results_data in model_label_results:

                    label_name = label_results_data["label"]
                    label_averages = label_results_data["averages"]
                    user_index = label_results_data["index"]

                    k = number_similar_profiles
                    Formatted_Model_Data = np.array(label_averages, dtype="float32")
                    numFeatures = len(label_averages[0])
                    index = faiss.IndexFlatL2(numFeatures)
                    index.add(Formatted_Model_Data)
                    D, I = index.search(Formatted_Model_Data, k)
                    label_similar_users_results = []

                    for SimilarRecordIndex in I:
                        for record_index in SimilarRecordIndex:
                            matching_username = user_index[record_index]
                            tmp_list = []
                            for record_index in SimilarRecordIndex:
                                profile_name = user_index[record_index]
                                if(profile_name is not matching_username):
                                    tmp_list.append(profile_name)
                            label_similar_users_results.append({"user": matching_username, "matching_users": tmp_list})
                           
                    Model_Response_Object.append({"label": label_name, "users": label_similar_users_results})
                Final_Response_Object.append({"model": model_name, "labels": Model_Response_Object})

            await self.db_engine.insert_similar_metric_records(Final_Response_Object)
            return Final_Response_Object
        
    async def formatUserSocialDataForIdentityComparisonAllModels(self, userData, timeFrame='30min', selected_periods=100):
        Formatted_User_Data_Array = {}
        Formatted_User_Data_Labels = []
        for record in userData:
                model_results = record["model_results"]
                record_text = record["text"]
                user_name = record["username"]
                dateTime = record["date_posted"]
                modelNum = len(model_results)
                staging_metric_list = []
                staging_metric_labels = []
                
                staging_metric_list.append(dateTime)
                staging_metric_labels.append('dateTime')
                
                for iterator1 in range(0, len(model_results)):
                    row = model_results[iterator1]
                    analysis = row["results"]
                    if(len(analysis) == 1):
                            analysis = analysis[0]
                            if(len(analysis) == 1):
                                analysis = analysis[0]
                    for iterator2 in range(0, len(analysis)):
                        scoreObject = analysis[iterator2]
                        try:
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            staging_metric_list.append(value) 
                            staging_metric_labels.append(label)
                        except Exception as e:
                            print("\nAn exception occurred when formatting social data")
                            print("\nanalysis Object:")
                            print(analysis)
                            print(e) 
                            
                    Formatted_User_Data_Labels.append(staging_metric_labels)
                    if(not user_name in Formatted_User_Data_Array):
                        Formatted_User_Data_Array[user_name] = [ [] for _ in range(modelNum) ]
                        Formatted_User_Data_Array[user_name][iterator1] = []
                        Formatted_User_Data_Array[user_name][iterator1].append({"text": record_text, "date" :dateTime, "value": value})
                        
                    else:
                        if(not Formatted_User_Data_Array[user_name][iterator1]):
                            Formatted_User_Data_Array[user_name][iterator1] = []
                            Formatted_User_Data_Array[user_name][iterator1].append({"text": record_text, "date" :dateTime, "value": value})
                            # Formatted_User_Data_Array[user_name][iterator1].append(staging_metric_list)
                            
                        else:
                            Formatted_User_Data_Array[user_name][iterator1].append({"text": record_text, "date" :dateTime, "value": value})
        
        Formatted_User_Averages_Array = []
        for username in Formatted_User_Data_Array:
            results = Formatted_User_Data_Array[username]
            Model_Averages_Array = []
            for model_result_array in results:
                try:
                    tmp_array = []
                    for record in model_result_array:
                        tmp_array.append([record["date"],record["value"]])

                    Identity_Df = np.array(tmp_array)
                    df = pd.DataFrame(Identity_Df, columns =  ["date","value"])
                    df.drop_duplicates(inplace=True)
                    df['date'] = pd.to_datetime(df['date'])
                    df['date'] = pd.DatetimeIndex(df['date'])
                    df.set_index('date').resample(timeFrame).mean().ffill().bfill()
                    # Only grab last x records
                    df.iloc[0:selected_periods]
                    del df[df.columns[0]]
                    df = df.astype(float)
                    model_results_resampled = df.to_numpy()
                    average_array = np.mean(model_results_resampled, axis=0)
                    Model_Averages_Array.append(average_array)

                except Exception as e:
                    print("\nAn exception occurred when formatting model_result_array")
                    print(e) 
            Formatted_User_Averages_Array.append({"username": username, "model_averages": Model_Averages_Array})
        user_comparison_array = []
        user_index_array = []
        for userObject in Formatted_User_Averages_Array:
            username = userObject["username"]
            model_averages = userObject["model_averages"]
            average_of_averages_array = np.mean(model_averages, axis=0)
            user_comparison_array.append(average_of_averages_array)
            user_index_array.append(username)

        for index, item in enumerate(user_comparison_array):
            if(user_index_array[index] == username):
                user_index_array.insert(0, username)
                user_index_array.pop(index)
                user_comparison_array.insert(0, user_comparison_array[index])
                user_comparison_array.pop(index)
            
        response_obj = {
            "user_averages": Formatted_User_Averages_Array,
            "user_comparison_array": user_comparison_array,
            "user_index": user_index_array
        }
        return response_obj

    async def findSimilarUsersAllModels(self, allUsers = True, username=False, number_similar_profiles=4, timeFrame='30min', selected_periods=100):
        userData = await self.db_engine.search_for_record('analysis_results')
        formatted_model_data = await self.formatUserSocialDataForIdentityComparisonAllModels(userData, timeFrame, selected_periods) 
        await self.db_engine.insert_similar_user_records(False, True)
        try:
            k = number_similar_profiles
            Similar_Results_Per_Model_Array = []
            model_index = formatted_model_data["user_index"]
            model_result_data = formatted_model_data["user_comparison_array"]
            Formatted_Model_Data = np.array(model_result_data, dtype="float32")
            numFeatures = len(Formatted_Model_Data[0])
            index = faiss.IndexFlatL2(numFeatures)
            index.add(Formatted_Model_Data)
            D, I = index.search(Formatted_Model_Data, k)
            model_results = []
            for SimilarRecordIndex in I:
                recordIndex = SimilarRecordIndex[0]
                selectedRecord = model_index[recordIndex]
                model_results.append(selectedRecord)
            Final_Response_Object = []
            for profileList in I:
                user_to_compare = model_results[profileList[0]]
                if(allUsers):
                    tmp_list = []
                    for similar_profile in profileList:
                        profile_name = model_results[similar_profile]
                        if(profile_name is not user_to_compare):
                            tmp_list.append( model_results[similar_profile])
                    Final_Response_Object.append({"user_searched": user_to_compare, "similar_Profiles":tmp_list})
                else:
                    if(user_to_compare == username):
                        tmp_list = []
                        for similar_profile in profileList:
                            profile_name = model_results[similar_profile]
                            if(profile_name is not user_to_compare):
                                tmp_list.append( model_results[similar_profile])
                        Final_Response_Object.append({"user_searched": username, "similar_Profiles":tmp_list})
            await self.db_engine.insert_similar_user_records(Final_Response_Object)
            return Final_Response_Object
        except Exception as e: 
            print(e)
    #================================            
    # Start Prediction           
    #================================    
    # 
    async def formatUserSocialDataProphet(self, userData, modelNumber):
        Formatted_Result_List_After_Analysis = []
        ListOfLabels = []
        Model_Name = False
        for iterator1 in range(1, len(userData)):
            row = userData[iterator1]
            model_results = row["model_results"]
            dateTimeValue = row["date_posted"]
            for result in model_results:
                if(result["modelId"] == modelNumber):
                    # print("\n\n",result, "\n")
                    analysis = result["results"]
                    returnValues = []
                    labels = []
                    returnValues.append(dateTimeValue)
                    labels.append('date_posted')
                    Model_Name = result["modelName"]
                    # print("analysis object length: ", len(analysis), "\n\n")
                    # if(len(analysis) == 1):
                    #     analysis = analysis[0]
                    for iterator2 in range(0, len(analysis)):
                        scoreObject = analysis[iterator2]
                        try:
                            label = scoreObject["label"]
                            value = scoreObject["score"]
                            labels.append(label)
                            returnValues.append(value)
                        except Exception as e:
                            print("An exception occurred when formatting social data")
                            print("analysis Object:")
                            print(analysis)
                            print(e)
                    Formatted_Result_List_After_Analysis.append(returnValues)
                    ListOfLabels.append(labels)
        # print(Formatted_Result_List_After_Analysis)
        if(len(ListOfLabels) == 0):
            return [[],[],""]
        print("List of Labels\n\n", ListOfLabels[0],"\n\n")
        return [ListOfLabels[0], Formatted_Result_List_After_Analysis, Model_Name]
       
    async def runProphetAnalysis(self, username, modelNumber, timeFrame='T30', selected_periods=100):
        userData = await self.db_engine.search_for_record('analysis_results', username)
        formatted_model_data = await self.formatUserSocialDataProphet(userData, modelNumber)
        base_path = os.path.expanduser(os.path.join('./results'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        user_path = os.path.expanduser(os.path.join(base_path,username))
        user_path_exist = os.path.exists(base_path)
        if(not user_path_exist):
            os.makedirs(user_path)
        try:
            Model_Labels = formatted_model_data[0]
            Model_Results = formatted_model_data[1]
            Model_Name = formatted_model_data[2]
            Identity_Df = np.array(Model_Results)
            df = pd.DataFrame(Identity_Df, columns = Model_Labels)
            df.drop_duplicates(inplace=True)
            df['date_posted'] = pd.to_datetime(df['date_posted'])
            df['date_posted'] = pd.DatetimeIndex(df['date_posted'])
            df_Minute = df.set_index('date_posted').resample(timeFrame).ffill().bfill()
            
            for label in Model_Labels:
                try:
                    if (label != 'date_posted'):
                        print("================================================")
                        print("Running Analytics for ",label,"\n")
                        print("================================================")
                        new_df = df[['date_posted',label]].copy()
                        new_df.rename(columns={"date_posted": "ds", new_df.columns[1]: "y"}, inplace = True)
                        new_df = new_df.astype({'y':'float'})
                        dataSize = new_df.size
                        m = NeuralProphet()
                        metrics = m.fit(df=new_df, freq=timeFrame)
                        future = m.make_future_dataframe(df=new_df, periods=selected_periods, n_historic_predictions=True)
                        future.tail()
                        forecast = m.predict(df=future)
                        plotTitle = username + " - " +   label 
                        # fig2.show()
                        user_path = os.path.expanduser(os.path.join(base_path, username))
                        base_path_exist = os.path.exists(base_path)
                        if(not base_path_exist):
                            os.makedirs(base_path)
                            print("Created base directory")
                        isExist = os.path.exists(base_path)
                        user_path_exist = os.path.exists(user_path)
                        if(not user_path_exist):
                            os.makedirs(user_path)
                            print("Created user directory")
                        
                        filename1 = user_path + "/" + username + "_" + Model_Name + "_" + label + "_" + "predictions.png" 
                        fig_param1 = m.plot(forecast, xlabel="date_posted", ylabel=plotTitle)
                        fig_param1.savefig(filename1)
                        
                        filename2 = user_path + "/" + username + "_" + Model_Name + "_" + label + "_" + "micro_trends.png" 
                        fig_param2 = m.plot_components(forecast)
                        fig_param2.savefig(filename2)
                        
                        filename3 = user_path + "/" + username + "_" + Model_Name + "_" + label + "_" + "seasonal_trends.png" 
                        fig_param3 = m.plot_parameters()
                        fig_param3.savefig(filename3)
                        
                        
                except Exception as e:
                    print(df)
                    print(e)
        except Exception as e:
            print("Error in runProphetAnalysis")
            print(e)
            
      