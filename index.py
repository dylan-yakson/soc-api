
import os
import re
import array
import json
import time
import csv
import random
import asyncio
import logging, sys, getopt
import multiprocessing as mp
from multiprocessing import Process, Lock
from datetime import date, datetime
from social_engine import Reddit, Twitter
from prediction_engine import Prediction_Engine
from db_engine import DB_Engine
from osint_engine import Osint_Engine
from crypto_engine import Crypto_Engine
from task_engine import Task_Scheduler
from graph_engine import Graphing_Engine
from nlp_models import getModels, getModelIds

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.disable(sys.maxsize)
today = date.today()
todays_date_formatted = today.strftime("%Y-%m-%d")

async def processFindSimilarUsersAllModels(prediction_engine, num_similar_profiles, time_interval_similar_users, num_intervals_to_analyze_similar_users, desired_metric_threshold):
    base_path = './results'
    base_path_exist = os.path.exists(base_path)
    if(not base_path_exist):
        os.makedirs(base_path)
        print("\nCreated base directory")
    Similar_Users = await prediction_engine.findSimilarUsersAllModels(True, False, num_similar_profiles, time_interval_similar_users, num_intervals_to_analyze_similar_users)
    similar_users_json_object = json.dumps(Similar_Users, indent=4)
    similar_users_json_filename = base_path + "/_SIMILAR_USERS.json"
    with open(similar_users_json_filename, "w") as outfile:
        outfile.write(similar_users_json_object)
        
async def processFindSimilarUsersPerModelLabel(prediction_engine,  num_similar_profiles, time_interval_similar_users, num_intervals_to_analyze_similar_users, desired_metric_threshold):
    base_path = './results'
    base_path_exist = os.path.exists(base_path)
    if(not base_path_exist):
        os.makedirs(base_path)
        print("\nCreated base directory")
    top_user_model_result_data = await prediction_engine.findSimilarUsersPerModelMetric(num_similar_profiles, time_interval_similar_users,    num_intervals_to_analyze_similar_users, desired_metric_threshold)
    similar_users_per_metric_json_object = json.dumps(top_user_model_result_data, indent=4)
    similar_users_per_metric_json_filename = base_path +  "/_SIMILAR_USERS_PER_MODEL_METRIC.json"
    with open(similar_users_per_metric_json_filename, "w") as outfile:
        outfile.write(similar_users_per_metric_json_object)    
    
def processSimilarUsersFromUsername(prediction_engine,num_similar_profiles, time_interval_similar_users,    num_intervals_to_analyze_similar_users, desired_metric_threshold):
    num_similar_profiles = 10
    time_interval_similar_users = "30T"
    num_intervals_to_analyze_similar_users = 96
    desired_metric_threshold = .5
    p = Process(target=asyncio.run(processFindSimilarUsersPerModelLabel(prediction_engine, num_similar_profiles, time_interval_similar_users, num_intervals_to_analyze_similar_users, desired_metric_threshold)))
    p.start()
    p = Process(target=asyncio.run(processFindSimilarUsersAllModels(prediction_engine, num_similar_profiles, time_interval_similar_users, num_intervals_to_analyze_similar_users, desired_metric_threshold)))
    p.start()
        
def processPredictionsFromUsername(prediction_engine, user_list, model_id_list, time_interval_predictions, num_intervals_to_predict):
    base_path = './results'
    base_path_exist = os.path.exists(base_path)
    if(not base_path_exist):
        os.makedirs(base_path)
        print("\nCreated base directory")
    for username in user_list:
        user_path = os.path.join(base_path, username)
        user_path_exist = os.path.exists(user_path)
        if(not user_path_exist):
            os.makedirs(user_path)
            print("\nCreated user directory")
        asyncio.run(prediction_engine.generateWordCloudForUser(username,['http','https', 't','co', 'link','newsalert','newsconnect']))
        # p = Process(target =asyncio.run(prediction_engine.generateWordCloudForUser(username,['http','https', 't','co', 'link','newsalert','newsconnect'])))
        # p.start()
        print(model_id_list)
        for model_id in model_id_list:
            # asyncio.run(prediction_engine.runProphetAnalysis(username, model_id, time_interval_predictions, num_intervals_to_predict))
            p = Process(target = asyncio.run(prediction_engine.runProphetAnalysis(username, model_id, time_interval_predictions, num_intervals_to_predict)))
            p.start()
        p.join()
            
def processUserList(User_List, prediction_engine, process_queue):
    for user_name in User_List:
        print(user_name)
        p = Process(target = asyncio.run(prediction_engine.runModelsOnSingleScrapedUser(user_name)))
        p.start()
    p.join()
        

def processScrapeTerm(searchTerm, db_engine, runModelsBeforeSave=True):
    twitter = Twitter(db_engine)
    p = Process(target = asyncio.run(twitter.Search_For_Search_Term(searchTerm)))
    p.start()
      
async def main(args):
    import asyncio

    flip_flag = False
    flip_inverted_flag = False
    
    process_flag = False
    query_flag = False
    db_backup_flag = False
    similar_users_flag = False
    predictions_flag = False
    create_db_flag = False
    create_image_flag = False
    find_similar_users_flag = False
    sync_database_flag = False
    ZIP_RESULTS_DIR = False
    test_flag= False
    install_dep_flag= False
    wait_flag=False
    wipe_db_flag=False
    no_analysis_flag = False
    create_users_flag = False
    scrape_reddit_flag = False
    scrape_twitter_flag = False
    scrape_twitter_fast_flag = False
    scrape_reddit_fast_flag = False
    restore_database_flag= False
    process_fast_flag = False
    graph_User_Flag = False
    # scrape_flag = False
    if args.install:
        install_dep_flag = True 
    elif args.scrapeTwitter:
        scrape_twitter_flag = args.scrapeTwitter
    elif args.scrapeTwitterFast:
        scrape_twitter_fast_flag = args.scrapeTwitterFast
    elif args.analyzeFast:
        process_fast_flag = args.analyzeFast
    elif args.scrapeRedditFast:
        scrape_reddit_fast_flag = args.scrapeRedditFast
    elif args.scrapeReddit:
        scrape_reddit_flag = args.scrapeReddit
    elif args.checkEmail:
        osint_engine = Osint_Engine()
        email_analysis = await osint_engine.email_analysis(args.checkEmail)
        for result in email_analysis:
            print(result)
    elif args.reverse:
        flip_flag = True
    elif args.analyze:
        process_flag = True
    elif args.wait:
        wait_flag = True
    elif args.genimg:
        create_image_flag = True
    elif args.query:
        query_flag = True
    elif args.simusers:
        find_similar_users_flag = True
    elif args.backup:
        db_backup_flag = True
    elif args.predmetric:
        predictions_flag = True
    elif args.zip:
        ZIP_RESULTS_DIR = True
    elif args.createdb:
        create_db_flag = True
    elif args.wipedb:
        wipe_db_flag = True
    elif args.syncdb:
        sync_database_flag = True  
    elif args.restoredb:
        restore_database_flag = args.restoredb  
    elif args.createusers:
        create_users_flag = args.createusers  
    elif args.runTask:
        await Task_Scheduler().scheduleTask(args.runTask)
    elif args.delTask:
        await Task_Scheduler().delTask(args.delTask)
    elif args.osintInstall:
        osint_engine = Osint_Engine()
        await osint_engine.install()
    elif args.graphUser:
        graph_User_Flag = args.graphUser  
        
    elif args.checkUsername:
        osint_engine = Osint_Engine()
        usernames = osint_engine.username_analysis(args.checkUsername, "")
    db_engine = None
    prediction_engine = None
    process_queue = mp.Queue()
    try:
        db_username = os.getenv('DB_USERNAME')
        db_pass = os.getenv('DB_PASSWORD')
        db_database = os.getenv('DB_DATABASE')
        db_server = os.getenv('DB_HOST')

        if(db_username is None or db_pass is None or db_database is not None or db_server is not None):
            db_engine = DB_Engine(db_username, db_pass, db_database, db_server)
            await db_engine.initializeConnection()
            await db_engine.checkEnv()
        db_engine = DB_Engine(db_username, db_pass, db_database, db_server)
        # await db_engine.createPostgresServer()
        await db_engine.initializeConnection()
        models = getModels()
        prediction_engine = Prediction_Engine(models, db_engine)
    except Exception as e:
        print("\n===============================================================================================================")
        print("\nYou probably didn't install shit ya fuck-WAD.... I gotchu tho... LIKE ALWAYS... don't fret G!")
        print("\n===============================================================================================================")
        print("\n'Real' error: ", e, "\n\n")

        print("\nError connecting to database... ")
        print("\n\n",e,"\n")
        db_username = str(os.getenv('DB_USERNAME')).lower()
        db_pass = os.getenv('DB_PASSWORD')
        db_database = str(os.getenv('DB_DATABASE')).lower()
        db_server = os.getenv('DB_HOST')

        db_engine = DB_Engine(db_username, db_pass, db_database, db_server)
        models = getModels()
        prediction_engine = Prediction_Engine(models, db_engine)
        await db_engine.checkEnv()
        await db_engine.initializeConnection()

    if args.runBasic:
            await Task_Scheduler().scheduleTask("-scrapeTwitterFast ukraine taiwan evil hate")
            await Task_Scheduler().scheduleTask("-scrapeRedditFast antiwork sino vent all")
            await Task_Scheduler().scheduleTask("-analyze 1")
            await Task_Scheduler().saveTasks()
    if(args.createdb):
            await db_engine.createPostgresServer()
            await db_engine.initializeConnection()

    if(args.wipedb):
        try:
            db_username = str(os.getenv('DB_USERNAME')).lower()
            db_pass = os.getenv('DB_PASSWORD')
            db_database = str(os.getenv('DB_DATABASE')).lower()
            db_server = os.getenv('DB_HOST')
            db_engine = DB_Engine(db_username, db_pass, db_database, db_server)
            await db_engine.wipe_db()
        except Exception as e:
                print("\nError removing database... You check the env file?")
                print(e)

    if(restore_database_flag):
        try:
            db_username = str(os.getenv('DB_USERNAME')).lower()
            db_pass = os.getenv('DB_PASSWORD')
            db_database = str(os.getenv('DB_DATABASE')).lower()
            db_server = os.getenv('DB_HOST')
            db_engine = DB_Engine(db_username, db_pass, db_database, db_server)
            await db_engine.restoreDatabaseFromBackup(restore_database_flag)
        except Exception as e:
                print("\nError removing database... You check the env file?")
                print(e)
    if(prediction_engine is not None and db_engine is not None):
        if(install_dep_flag):
            try:
                await db_engine.initializeConnection()
                await Task_Scheduler().install()
            except Exception as e:
                print("\nError initializing connection...")
                print(e)
                await db_engine.createPostgresServer()
            try:
                await Task_Scheduler().install()
            except Exception as e:
                print("\nError installing task scheduler...")
                print(e)
        
        if(db_backup_flag):
            print("\nSyncing Database")
            await db_engine.backupPostgresServer()

        if(create_users_flag):
            # Initialize Reddit Engine
            reddit_engine = Reddit(db_engine)
            # Generate Reddit Account
            reddit_engine.Create_Account(int(create_users_flag))

            # # Initialize Twitter Engine
            # twitter_engine = Twitter(db_engine)
            # # Generate Twitter Account
            # twitter_engine.Create_Account(int(create_users_flag))
        if(graph_User_Flag):
            await prediction_engine.create_social_graph(graph_User_Flag)
        if(scrape_reddit_flag):
            # Pull random generated reddit account
            reddit_engine = Reddit(db_engine)
            reddit_user = reddit_engine.pullAccount()

            print("Reddit User", reddit_user)
            # To search for reddit users in sub
            for term in scrape_reddit_flag:
                # await reddit_engine.pullUsersFromSub(term[0])
                p = Process(target=asyncio.run(reddit_engine.pullUsersFromSub(term[0])))
                p.start()
            p.join()

        if(scrape_reddit_fast_flag):
            # Pull random generated reddit account
            reddit_engine = Reddit(db_engine)
            reddit_user = reddit_engine.pullAccount()
            print("Reddit User", reddit_user)
            for term in scrape_reddit_fast_flag:
                p = Process(target=asyncio.run(reddit_engine.pullUsersFromSubFast(term[0])))
                p.start()
            p.join()

        if(scrape_twitter_flag):
            searchTerms = scrape_twitter_flag 
            for searchTerm in searchTerms:
                p = Process(processScrapeTerm(searchTerm, db_engine))
                p.start()
                p.join()

        if(scrape_twitter_fast_flag):
            searchTerms = scrape_twitter_fast_flag 
            print(searchTerms)
            for searchTerm in searchTerms:
                p = Process(processScrapeTerm(searchTerm, db_engine))
                p.start()
            p.join()

        if(process_flag):
            Users_List = await prediction_engine.pullListOfUsers('sociality')
            print(Users_List)
            splitedSize = 15
            a_splited = [Users_List[x:x+splitedSize] for x in range(0, len(Users_List), splitedSize)]
            for split_users in a_splited:
                #processUserList(split_users, prediction_engine, process_queue)
                p = Process(target =processUserList, args=(split_users, prediction_engine, process_queue))
                p.start()
                p.join()
            
        if(process_fast_flag):
            Users_List = await prediction_engine.pullListOfUsers('sociality')
            print(Users_List)
            splitedSize = 15
            a_splited = [Users_List[x:x+splitedSize] for x in range(0, len(Users_List), splitedSize)]
            for split_users in a_splited:
                #processUserList(split_users, prediction_engine, process_queue)
                p = Process(target =processUserList, args=(split_users, prediction_engine, process_queue))
                p.start()
            p.join()
            
        if(query_flag):
            query_results = await db_engine.run_custom_query("SELECT COUNT(*) FROM sociality")
            print("\n\n\nScraped Social Media Records: ",query_results)
            query_results = await db_engine.run_custom_query("SELECT COUNT(*) FROM analysis_results")
            print("\n\n\nProcessed Social Media Records: ",query_results)
            query_results = await db_engine.run_custom_query("SELECT COUNT(*) FROM similar_users")
            print("\n\n\nModels - Similar Users: ",query_results)
            query_results = await db_engine.run_custom_query("SELECT COUNT(*) FROM metric_user_similarities")
            print("\n\n\nModel Metrics - Similar Users: ",query_results)
            query_results = await db_engine.run_custom_query("SELECT COUNT(*) FROM model_predictions")
            print("\n\n\nProcessed Social Model Predictions: ",query_results)
            
            # query_results = await db_engine.run_custom_query("SELECT * FROM metric_user_similarities")
            # print("\n\n\nmetric_user_similarities:\n\n ",query_results[::-1][0:5])


        if(predictions_flag):
            users_list = await prediction_engine.pullListOfUsers('analysis_results')
            splitedSize = 15 #round(len(users_list) / 10)
            a_splited = [users_list[x:x+splitedSize] for x in range(0, len(users_list), splitedSize)]
            for split_users in a_splited:
                model_id_list = await getModelIds()
                time_interval_predictions = "30T"
                num_intervals_to_predict = 5
                time_interval_similar_users = "30T"
                num_intervals_to_analyze_similar_users = 5
                num_similar_profiles = 5
                p = Process(target = processPredictionsFromUsername , args=(prediction_engine, split_users,model_id_list, time_interval_predictions, num_intervals_to_predict))
                p.start()

        if(create_image_flag == True):
            Users_List=[]
            Users_List = await prediction_engine.pullListOfUsers('analysis_results')
            base_path = './results'
            base_path_exist = os.path.exists(base_path)
            if(not base_path_exist):
                os.makedirs(base_path)
                print("\nCreated base directory")
            for username in Users_List:
                user_path = os.path.expanduser(os.path.join(base_path, username))
                user_path_exist = os.path.exists(user_path)
                if(not user_path_exist):
                    os.makedirs(user_path)
                    print("\nCreated user directory")
                try:
                    # ==================================
                    #  Create GIF for image progression per model
                    # ==================================
                    try:
                        images2 = await prediction_engine.createGIFFromSocialModelData(username)
                    except Exception as e:
                        print(e)
                    # ==================================
                    #  Create images per model
                    # ==================================
                    images = await prediction_engine.createImageFromSocialModelData(username)
                    print("\nImages: ", images)
                    # ==================================
                    #  Concat images into one big image & show images for all models individually as well
                    # ==================================
                    def concat_Image_horizontal(im1, im2):
                        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
                        dst.paste(im1, (0, 0))
                        dst.paste(im2, (im1.width, 0))
                        return dst

                    concatenated_image = False
                    for image in images:
                        print(image)
                        model_name = image["model"]
                        model_image = image["image"]
                        single_model_img_path = user_path + "/" + username + "_IMAGE_" +model_name +"_.png" 
                        model_image.save(single_model_img_path)
                        try:
                            if(concatenated_image == False):
                                concatenated_image = model_image
                            else:
                                concatenated_image = concat_Image_horizontal(concatenated_image,model_image)
                            concat_img_path = user_path + "/" + username + "_IMAGE_ALL_MODELS.png" 
                            concatenated_image.save(concat_img_path)
                        except Exception as e:
                            print("\nException when generating image for user")
                            print(e)
                except Exception as e:
                    print("\nException when generating image for user")
                    print(e)
                    
        if(find_similar_users_flag == True):
            base_path = './results'
            base_path_exist = os.path.exists(base_path)
            if(not base_path_exist):
                os.makedirs(base_path)
                print("\nCreated base directory")
            num_similar_profiles = 10
            time_interval_similar_users = "30T"
            num_intervals_to_analyze_similar_users = 96
            desired_metric_threshold = .3
            p = Process(target = processSimilarUsersFromUsername , args=(prediction_engine, num_similar_profiles, time_interval_similar_users,    num_intervals_to_analyze_similar_users, desired_metric_threshold))
            p.start()

        if(sync_database_flag == True):
            await db_engine.syncFileWithPostgres('./records_to_sync.json')

        if(ZIP_RESULTS_DIR == True):
            p = Process(target=shutil.make_archive('./predictions', 'zip', './results'))
            p.start()
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-install', help='Install dependencies')
    parser.add_argument('-createdb', help='Setup Postgres Database')
    parser.add_argument('-scrapeReddit', action='append' ,nargs='+', help='Scrape Reddit users for particular sub')
    parser.add_argument('-scrapeRedditFast', action='append' ,nargs='+', help='Scrape Reddit users for particular sub.... All at once... Really fast... Crash yo shit fast.')
    parser.add_argument('-scrapeTwitter', action='append' ,nargs='+', help='Scrape Twitter for keywords')
    parser.add_argument('-scrapeTwitterFast', action='append' ,nargs='+', help='Scrape Twitter for keywords.... All at once... Really fast... Crash yo shit fast.')
    parser.add_argument('-noprocess', help='Scrape Text only - Do not run models')
    parser.add_argument('-analyze', help='Run analysis on scraped records that havent been processed')
    parser.add_argument('-analyzeFast', help='Run analysis on scraped records that havent been processed.... All at once... Really fast... Crash yo shit fast.')
    parser.add_argument('-genimg', help='Generate Image for NLP data')
    parser.add_argument('-query', help='Run custom query on database records')
    parser.add_argument('-simusers', help='Find similar users based on model metric')
    parser.add_argument('-backup', help='Backup database to folder at cwd')
    parser.add_argument('-predmetric', help='Predict psychometrics from past results')
    parser.add_argument('-zip', help='Compress backup at cwd to zip')
    parser.add_argument('-syncdb', help='Populate your db from a json file')
    parser.add_argument('-wait', help='Wait for Processs to finish before starting a new batch of users... lol ')
    parser.add_argument('-wipedb', help='wipe that bitch')
    parser.add_argument('-restoredb', help='restore that bitch')
    parser.add_argument('-reverse', help='Deprecated: reverses processing order to speed up processing.. ')
    parser.add_argument('-runTask', help='schedule a task in pm2')
    parser.add_argument('-delTask', help='delete a scheduled task in pm2')
    parser.add_argument('-runBasic', help='Run thing that is standardized from the thing')
    parser.add_argument('-createusers', help='Create Users for social media profiles')
    parser.add_argument('-osintInstall', help='Install OSINT modules')
    parser.add_argument('-checkUsername', help='Finds Similar Usernames')
    parser.add_argument('-checkDomain', help='Performs Recon on domain')
    parser.add_argument('-checkEmail', help='Performs Recon on email')
    parser.add_argument('-checkPhone', help='Performs Recon on Phone Numbers')
    parser.add_argument('-detectPlates', help='Detects license plates from image path')
    parser.add_argument('-checkPlates', help='(Not Finished) Expands info from license plate')
    parser.add_argument('-trainModels', help='trains OSINT models')
    parser.add_argument('-graphPcap', help='Graphs a PCAP file in networkX')
    parser.add_argument('-graphTraffic', help='Graphs current network traffic on all interfaces for specified number of packets')
    parser.add_argument('-graphUser', help='(Not Finished) Graphs users social interactions in networkX')

    args = parser.parse_args()

    if(args.simusers or args.osintInstall): # or args.analyze):
        mp.set_start_method('spawn')

    # OSINT stuff (not async yet is why)
    if args.checkUsername:
        osint_engine = Osint_Engine()
        osint_engine.username_analysis(args.checkUsername, "")
    
    elif args.checkDomain:
        osint_engine = Osint_Engine()
        domain_analysis = osint_engine.domain_analysis(args.checkDomain)
        print("Domain Analysis:\n ",domain_analysis)

    elif args.checkPhone:
        osint_engine = Osint_Engine()
        results_path = "./{0}_results_tmp.txt".format(args.checkPhone)
        phone_results = osint_engine.phone_analysis(args.checkPhone, results_path)
        phone_results = ""
        with open(results_path, 'r') as results:
            phone_results = results.read()
        os.remove(results_path) 
        print("Phone Results:", phone_results)

    elif args.graphPcap:
        print(args.graphPcap)
        pcap_path = args.graphPcap
        graph_engine = Graphing_Engine("network", [])
        graph_engine.graph_pcap(pcap_path)

    elif args.graphTraffic:
        print(args.graphTraffic)
        try:
            num_packets = int(args.graphTraffic)
        except Exception as e:
            print("Error parsing int - setting to 1000")
            print(e)
            num_packets = 1000
        graph_engine = Graphing_Engine("network", [])
        graph_engine.graph_network_traffic(num_packets, "eth0")

    elif args.detectPlates:
        osint_engine = Osint_Engine()
        found_plates = osint_engine.detect_plates_from_image(args.detectPlates)
        print("Licence Plates Found: {0}".format(found_plates))

    elif args.trainModels:
        osint_engine = Osint_Engine()
        osint_engine.train_models()
    else:
        asyncio.run(main(args))