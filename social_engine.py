import urllib.request
import random
import json
import time
import csv
import logging, sys, getopt
import os
import random
import array
from datetime import date, datetime
from multiprocessing import Process
from random import randint
import asyncio
import re
import pickle 
from data_generator import generateServerPassword, generateUsername
from nlp_models import pullEntitiesFromText

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
    from dotenv import load_dotenv
    load_dotenv('./.env')
try:
    import asyncpg
except Exception as e:
    print("\ninstalling asyncpg")
    print(os.system("pip install asyncpg"))
    import asyncpg

try:
    import numpy as np
except Exception as e:
    print("\ninstalling numpy")
    print(os.system("pip install numpy"))
    import numpy as np

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
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
try:
    import pandas as pd
except Exception as e:
    print("\ninstalling pandas")
    print(os.system("pip install pandas"))
    import pandas as pd

try:
    import multiprocessing as mp
except Exception as e:
    print("\ninstalling multiprocessing")
    print(os.system("pip install multiprocessing"))
    import multiprocessing as mp

try:
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
except Exception as e:
    print("\ninstalling plotly")
    print(os.system("pip install plotly"))
    print(os.system("pip install matplotlib"))
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

try:
    import twint
except Exception as e:
    print("\ninstalling twint")
    os.system("pip install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint")
    import twint

try:
    import praw
except Exception as e:
    print("\ninstalling praw")
    os.system("pip install praw")
    import praw

try:
    import faiss
except Exception as e:
    print("\ninstalling faiss")
    print(os.system("pip install faiss-gpu"))
    import faiss

try:
    from sklearn.svm import SVR
except Exception as e:
    print("\ninstalling scikit-learn")
    print(os.system("pip install scikit-learn"))
    from sklearn.svm import SVR


try:
    from neuralprophet import NeuralProphet, set_log_level
except Exception as e:
    print("\ninstalling neuralprophet")
    print(os.system("pip install neuralprophet"))
    from neuralprophet import NeuralProphet, set_log_level


try:
    from scipy.special import softmax
except Exception as e:
    print("\ninstalling scipy")
    print(os.system("pip install scipy"))
    from scipy.special import softmax

try:
    from faker import Faker
except Exception as e:
    print("\ninstalling faker")
    print(os.system("pip install faker"))
    from faker import Faker

try:
    from tinydb import TinyDB, Query 
except Exception as e:
    print("\ninstalling tinydb")
    print(os.system("pip install tinydb"))
    from tinydb import TinyDB, Query 

try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("\ninstalling pillow")
    print(os.system("pip install --upgrade protobuf==3.20.0"))
    print(os.system("pip install pillow"))
    from PIL import Image, ImageDraw

try:
    from wordcloud import WordCloud, STOPWORDS
except Exception as e:
    print("\ninstalling top2vec")
    print(os.system("pip install top2vec"))
    from wordcloud import WordCloud, STOPWORDS

try:
    from detoxify import Detoxify
except Exception as e:
    print("\ninstalling detoxify")
    print(os.system("pip install detoxify"))
    from detoxify import Detoxify

try:
    import nest_asyncio
    nest_asyncio.apply()

except Exception as e:
    print("\ninstalling detoxify")
    print(os.system("pip install nest_asyncio"))
    import nest_asyncio
    nest_asyncio.apply()

# ===============================================================================================================
#                                   Social Media Scraping Services
# ===============================================================================================================
#
class Reddit():
    def __init__(self, db_engine, username="", password="", client_id="", client_secret=""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.password = password
        self.username = username
        self.db_engine = db_engine
        self.site_name = "reddit"
        self.site_id = 0
        self.site_url = "https://reddit.com"

    def pullAccount(self):
        created = open('reddit_accounts.txt', 'r')
        account_list = created.read().split("\n")
        a_formatted = [x.split(":") for x in account_list[0:len(account_list) -1]] 
        a_formatted = [ {"user": x[0], "pass": x[1], "client_id": x[2], "secret": x[3] } for x in a_formatted]
        current_profile = a_formatted[randint(0, len(a_formatted) - 1)]
        self.client_id = current_profile["client_id"]
        self.client_secret = current_profile["secret"]
        self.password = current_profile["pass"]
        self.username = current_profile["user"]
        # print(current_profile)
        created.close()
        return current_profile

    # Creates a reddit account 
    # Then Creates a "script" app to use with praw
    # Then saves the app id, secret, username, and pass to a file to use later :)
    def Create_Account_Selenium_Task(self, username, password):
        argumentString = "user-data-dir=selenium" + username
        chrome_options = Options()
        chrome_options.add_argument(argumentString) 
        browser = webdriver.Chrome(ChromeDriverManager().install())
        #get reddit account creation page
        browser.set_window_size(683, 744)
        browser.get('http://old.reddit.com/login')
        #insert username
        time.sleep(randint(1,4))
        browser.find_element("id",'user_reg').click()
        browser.find_element("id",'user_reg').send_keys(username)
        #insert password
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd_reg').click()
        browser.find_element("id",'passwd_reg').send_keys(password)
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd2_reg').click()
        browser.find_element("id",'passwd2_reg').send_keys(password)

        #pause to manually enter captcha
        readytomove = False
        while not readytomove:
            captchaResponse = input("[*] Solve captcha, create account, then press Y... enter 'N' as input if captcha doesn't appear to skip username" + '\n')

            if (captchaResponse == 'N'):
                browser.quit()
                return False
            elif (captchaResponse == "Y"):
                browser.find_element(By.LINK_TEXT,'preferences').click()
                time.sleep(random.randint(1, 2))
                browser.find_element(By.LINK_TEXT,'apps').click()
                time.sleep(random.randint(1, 2))
                browser.find_element(By.ID,'create-app-button').click()
                time.sleep(random.randint(1, 2))
                browser.find_element(By.NAME,'name').click()
                browser.find_element(By.NAME,'name').send_keys("test1")

                browser.find_element(By.NAME,'redirect_uri').click()
                browser.find_element(By.NAME,'redirect_uri').send_keys("http://google.com")

                browser.find_element(By.ID,'app_type_script').click()

                browser.find_element(By.XPATH,'//*[@id="create-app"]/button').click()
                time.sleep(random.randint(1, 2))

                client_id = browser.find_element(By.XPATH,'//*[@class="app-details"]/h3[2]').text
                print("\nCLIENT_ID: {0}".format(client_id))
                secret = browser.find_element(By.XPATH,'//*[@class="edit-app-form"]/form/table/tbody/tr[1]/td').text
                print("SECRET: {0}".format(secret))
                act_verified = True
                return_string = "{0}:{1}:{2}:{3}".format(username,password,client_id,secret)
                created = open('reddit_accounts.txt', 'a')
                print("[+] writing {} to reddit_accounts.txt...".format(return_string))
                created.write(return_string)
                created.close()
                browser.quit()
                return return_string + '\n'

    def Create_Account(self, numberOfAccountsToGenerate=5):
        # os.system('clear')
        #run account generator for each user in list
        accNo = numberOfAccountsToGenerate
        i = 0

        while accNo > i:
            Faker.seed(random.randint(1,50))
            fake = Faker()
            name = fake.domain_word()
            print(name)
            name = name.replace(r'[\n\t\ ]+', ' ')  # t is your original text
            namesArr = re.findall('[A-Za-z]*', name)
            finalNameConcat = ""
            for record in namesArr:
                finalNameConcat+=record
            full_name = (finalNameConcat)
            username = full_name + str(random.randint(1000, 50000))
            password = generateServerPassword()
            cred = password
            print('[+] creating account for %s with password %s' % (username,password))
            account_created = self.Create_Account_Selenium_Task(username, password)
            print("\nAccount Created :\n", account_created)
            # os.system('service tor restart')
            if(account_created):
                i = i+1
            else:
                print('[-] name not recorded due to captcha issue')
            time.sleep(2)
    
    async def processRow(self, username,record, record_type="comment"):
        if(record_type == "comment"):
            try:
                record_data = {"createdDate":  record.created_utc, "parent_id": record.parent_id, "comment_id": record.id, "comment_body": record.body, "subreddit_id": record.subreddit_id, "parent_text": record.submission.selftext, "parent_name": record.submission.name}
                recordDate = datetime.utcfromtimestamp(record.created_utc)
                date_formatted = datetime.strftime(recordDate, '%Y-%m-%d %H:%M:%S')
                record_To_Insert = {
                    "username": username,
                    "messageId": record.id,
                    "text" : record.body,
                    "source": "reddit",
                    "dataType": "comment",
                    "data": record_data,
                    "dateTime": date_formatted,
                }
                await self.db_engine.insert_social_record(record_To_Insert)
            except Exception as e:
                print("\nException saving model result from Reddit comment")
                print(e)
        elif(record_type == "submission"):
            try:
                record_data = {"submission_text": record.selftext, "submission_id": record.id, "createdDate": record.created_utc, "subreddit": record.subreddit.display_name, "name": record.name, "num_comments": record.num_comments, "permalink": record.permalink, "title": record.title, "upvote_ratio": record.upvote_ratio}
                recordDate = datetime.utcfromtimestamp(record.created_utc)
                date_formatted = datetime.strftime(recordDate, '%Y-%m-%d %H:%M:%S')
                record_To_Insert = {
                    "username": username,
                    "messageId": record.id,
                    "text" : record.selftext,
                    "source": "reddit",
                    "dataType": "submission",
                    "data": record_data,
                    "dateTime": date_formatted,
                }
                await self.db_engine.insert_social_record(record_To_Insert)
            except Exception as e:
                print("\nException saving model result from Reddit Submission")
                print(e)
    
    async def pullUsersFromSub(self, subreddit="all"):
        print(subreddit)
        Faker.seed(random.randint(1,50))
        fake = Faker()
        useragent = fake.user_agent()
        #https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
        reddit = praw.Reddit(user_agent=useragent, client_id=self.client_id, client_secret=self.client_secret, username=self.username, password=self.password)
        comments, downvotes, hidden_content, saved_content, submissions = [[],[],[],[],[]]

        for submission in reddit.subreddit(subreddit).hot(limit=None):
            print("Checking for {0}".format(submission.author.name))
            p = Process(target =asyncio.run(self.Search_For_User(submission.author.name)))
            p.start()
            p.join()

    async def pullUsersFromSubFast(self, subreddit="all"):
            print(subreddit)
            Faker.seed(random.randint(1,50))
            fake = Faker()
            useragent = fake.user_agent()
            #https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
            reddit = praw.Reddit(user_agent=useragent, client_id=self.client_id, client_secret=self.client_secret, username=self.username, password=self.password)
            comments, downvotes, hidden_content, saved_content, submissions = [[],[],[],[],[]]

            for submission in reddit.subreddit(subreddit).hot(limit=None):
                print("Checking for {0}".format(submission.author.name))
                p = Process(target =asyncio.run(self.Search_For_User(submission.author.name)))
                p.start()
            p.join()
            
    async def Search_For_User(self, usernameToSearch):
        Faker.seed(random.randint(1,50))
        fake = Faker()
        useragent = fake.user_agent()
        #https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
        reddit = praw.Reddit(user_agent=useragent, client_id=self.client_id, client_secret=self.client_secret, username=self.username, password=self.password)
        comments, downvotes, hidden_content, saved_content, submissions = [[],[],[],[],[]]
        try:
            for comment in reddit.redditor(usernameToSearch).comments.new(limit=None):
                print(comment.body)
                print(comment.created_utc)
                insertedRecord = await self.db_engine.search_for_record("sociality", False, comment.id)
                if(not insertedRecord):
                    await self.processRow(usernameToSearch, comment, "comment" )
        except Exception as e:
            print("Error pulling comments")
            print(e)
        try:
            for submission in reddit.redditor(usernameToSearch).submissions.top(time_filter="year",limit=None):
                print(submission.selftext)
                insertedRecord = await self.db_engine.search_for_record('sociality', False, submission.id)
                if(not insertedRecord):
                    try:

                        await self.processRow(usernameToSearch, submission, "submission" )
                    except Exception as e:
                        print("\nException saving model result from Reddit comment")
                        print(e)
        except Exception as e:
            print("Error pulling submissions")
            print(e)

            
class Twitter():
    def __init__(self, db_engine, username="", password="", client_id="", client_secret=""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.password = password
        self.username = username
        self.db_engine = db_engine
        self.site_name = "twitter"
        self.site_id = 1
        self.site_url = "https://twitter.com"

    def pullAccounts(self):
        created = open('reddit_accounts.txt', 'r')
        account_list = created.read().split("\n")
        a_formatted = [x.split(":") for x in account_list[0:len(account_list) -1]] 
        a_formatted = [  ({"user": x[0], "pass": x[1], "client_id": x[2], "secret": x[3] } ) for x in a_formatted]
        # print(a_formatted)
        created.close()
        return a_formatted

    def Create_Account_Selenium_Task(self, username, password):
        argumentString = "user-data-dir=selenium" + username
        chrome_options = Options()
        chrome_options.add_argument(argumentString) 
        browser = webdriver.Chrome(ChromeDriverManager().install())
        #get reddit account creation page
        browser.set_window_size(683, 744)
        browser.get('http://old.reddit.com/login')
        #insert username
        time.sleep(randint(1,4))
        browser.find_element("id",'user_reg').click()
        browser.find_element("id",'user_reg').send_keys(username)
        #insert password
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd_reg').click()
        browser.find_element("id",'passwd_reg').send_keys(password)
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd2_reg').click()
        browser.find_element("id",'passwd2_reg').send_keys(password)

        #pause to manually enter captcha
        #TODO: complete captcha with audio or pic via open-cv
        readytomove = False
        while not readytomove:
            captchaResponse = input("[*] Solve captcha, create account, then press Y... enter 'N' as input if captcha doesn't appear to skip username" + '\n')

            if (captchaResponse == 'N'):
                browser.quit()
                return False
            elif (captchaResponse == "Y"):
                browser.find_element(By.LINK_TEXT,'preferences').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.LINK_TEXT,'apps').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.ID,'create-app-button').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.NAME,'name').click()
                browser.find_element(By.NAME,'name').send_keys("test1")

                browser.find_element(By.NAME,'redirect_uri').click()
                browser.find_element(By.NAME,'redirect_uri').send_keys("http://google.com")

                browser.find_element(By.XPATH,'//*[@id="create-app"]/button').click()
                time.sleep(random.randint(1, 2))

                client_id = browser.find_element(By.XPATH,'//*[@class="app-details"]/h3[2]').text
                secret = browser.find_element(By.XPATH,'//*[@class="edit-app-form"]/form/table/tbody/tr[1]/td').text
                return_string = "{0}:{1}:{2}:{3}".format(username,password,client_id,secret)
                created = open('reddit_accounts.txt', 'a')
                print("[+] writing {} to reddit_accounts.txt...".format(return_string))
                created.write(return_string)
                created.close()
                browser.quit()
                return return_string + '\n'
            
    def Create_Account(self, numberOfAccountsToGenerate=5):
        #run account generator for each user in list
        accNo = numberOfAccountsToGenerate
        i = 0

        while accNo > i:
            Faker.seed(random.randint(1,50))
            fake = Faker()
            name = fake.domain_word()
            print(name)
            name = name.replace(r'[\n\t\ ]+', ' ')  # t is your original text
            namesArr = re.findall('[A-Za-z]*', name)
            finalNameConcat = ""
            for record in namesArr:
                finalNameConcat+=record
            full_name = (finalNameConcat)
            username = full_name + str(random.randint(1000, 50000))
            password = generateServerPassword()
            cred = password
            print('[+] creating account for %s with password %s' % (username,password))
            account_created = self.Create_Account_Selenium_Task(username, password)
            print("\nAccount Created :\n", account_created)
            # os.system('service tor restart')
            if(account_created):
                i = i+1
            else:
                print('[-] name not recorded due to captcha issue')
            time.sleep(2)
    
    async def processRow(self, username, row):
        try:
            messageId = row["id"]
            insertedRecord = await self.db_engine.search_for_record('sociality', False, messageId)
            if(not insertedRecord):
                tweet = row["tweet"]
                print("\n================================================")
                print("\nChecking Tweet",tweet,"\n")
                print("\n================================================")
                result = row.to_json(orient="columns")
                parsed = json.loads(result)
                row_data = json.dumps(parsed, indent=4) 
                record_To_Insert = {
                    "username": username,
                    "messageId": messageId,
                    "text" : tweet,
                    "source": "twitter",
                    "dataType": "submission",
                    "data": row_data,
                    "dateTime": row['date'],
                }
                record_To_Insert["model_results"] ={}
                record_To_Insert["entities_extracted"] ={}
                await self.db_engine.insert_social_record(record_To_Insert)
        except Exception as e:
            print("\nAn exception occurred when running NLP model on tweets")
            print(e)

    async def Search_For_User(self, username, since='2018-01-01', until=todays_date_formatted, verbose = False ):
        try:
            b = twint.Config()
            b.Username = username
            b.Pandas = True
            # b.Limit = 90
            b.Hide_output = True
            b.Since = since
            b.until= until
            twint.run.Search(b)
            Identity_Tweets_List = twint.storage.panda.Tweets_df
            Identity_Data = {"identity": b.Username, "tweets": Identity_Tweets_List}
            Identity_Tweets_List = Identity_Tweets_List.reset_index()
            today = date.today()
            todays_date_formatted = today.strftime('%Y-%m-%d %H:%M:%S')
            for index, row in Identity_Tweets_List.iterrows():
                try:
                    await self.processRow(username, row)
                except Exception as e:
                    print("\nAn exception occurred when pulling tweets")
                    print(e)
        except Exception as e:
                print("\nAn exception occurred when pulling tweets")
                print(e)
                
    async def Search_For_Search_Term(self, searchTerm, no_analysis_flag=True, since='2018-01-01', until=todays_date_formatted, verbose = False ):
        try:        
            b = twint.Config()
            b.Search = searchTerm
            b.Pandas = True
            b.Limit = 90
            b.Hide_output = True
            b.Since = since
            b.until= until
            twint.run.Search(b)
            Identity_Tweets_List = twint.storage.panda.Tweets_df
            Identity_Tweets_List = Identity_Tweets_List.reset_index()  # make sure indexes pair with number of rows
            today = date.today()
            todays_date_formatted = today.strftime('%Y-%m-%d %H:%M:%S')
            t3 = False
            for index, row in Identity_Tweets_List.iterrows():
                author = row["username"]
                print(author)
                try:
                    t3 = Process(target=asyncio.run(self.Search_For_User(author)))
                    t3.start()
                    # t3.join()
                except Exception as e:
                    print(e)
            if(t3):
                t3.join()
        except Exception as e:
            print("\nAn exception occurred when pulling tweets")
            print(e)

class LinkedIn():
    # Not done yet
    def __init__(self, db_engine, username="", password="", client_id="", client_secret=""):
        self.client_id = client_id
        self.client_secret = client_secret
        self.password = password
        self.username = username
        self.db_engine = db_engine
        self.site_name = "twitter"
        self.site_id = 1
        self.site_url = "https://twitter.com"

    def pullAccounts(self):
        # Move created accounts to db? or keep them locally?
        created = open('linkedin_accounts.txt', 'r')
        account_list = created.read().split("\n")
        a_formatted = [x.split(":") for x in account_list[0:len(account_list) -1]] 
        a_formatted = [  ({"user": x[0], "pass": x[1], "client_id": x[2], "secret": x[3] } ) for x in a_formatted]
        # print(a_formatted)
        created.close()
        return a_formatted

    def Create_Account_Selenium_Task(self, username, password):
        argumentString = "user-data-dir=selenium" + username
        chrome_options = Options()
        chrome_options.add_argument(argumentString) 
        browser = webdriver.Chrome(ChromeDriverManager().install())
        #get reddit account creation page
        browser.set_window_size(683, 744)
        browser.get('http://old.reddit.com/login')
        #insert username
        time.sleep(randint(1,4))
        browser.find_element("id",'user_reg').click()
        browser.find_element("id",'user_reg').send_keys(username)
        #insert password
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd_reg').click()
        browser.find_element("id",'passwd_reg').send_keys(password)
        time.sleep(randint(1,5))
        browser.find_element("id",'passwd2_reg').click()
        browser.find_element("id",'passwd2_reg').send_keys(password)

        #pause to manually enter captcha
        #TODO: complete captcha with audio or pic via open-cv
        readytomove = False
        while not readytomove:
            captchaResponse = input("[*] Solve captcha, create account, then press Y... enter 'N' as input if captcha doesn't appear to skip username" + '\n')

            if (captchaResponse == 'N'):
                browser.quit()
                return False
            elif (captchaResponse == "Y"):
                browser.find_element(By.LINK_TEXT,'preferences').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.LINK_TEXT,'apps').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.ID,'create-app-button').click()
                time.sleep(random.randint(1, 3))
                browser.find_element(By.NAME,'name').click()
                browser.find_element(By.NAME,'name').send_keys("test1")

                browser.find_element(By.NAME,'redirect_uri').click()
                browser.find_element(By.NAME,'redirect_uri').send_keys("http://google.com")

                browser.find_element(By.XPATH,'//*[@id="create-app"]/button').click()
                time.sleep(random.randint(1, 2))

                client_id = browser.find_element(By.XPATH,'//*[@class="app-details"]/h3[2]').text
                secret = browser.find_element(By.XPATH,'//*[@class="edit-app-form"]/form/table/tbody/tr[1]/td').text
                return_string = "{0}:{1}:{2}:{3}".format(username,password,client_id,secret)
                created = open('reddit_accounts.txt', 'a')
                print("[+] writing {} to reddit_accounts.txt...".format(return_string))
                created.write(return_string)
                created.close()
                browser.quit()
                return return_string + '\n'
            
    def Create_Account(self, numberOfAccountsToGenerate=5):
        #run account generator for each user in list
        accNo = numberOfAccountsToGenerate
        i = 0

        while accNo > i:
            Faker.seed(random.randint(1,50))
            fake = Faker()
            name = fake.domain_word()
            print(name)
            name = name.replace(r'[\n\t\ ]+', ' ')  # t is your original text
            namesArr = re.findall('[A-Za-z]*', name)
            finalNameConcat = ""
            for record in namesArr:
                finalNameConcat+=record
            full_name = (finalNameConcat)
            username = full_name + str(random.randint(1000, 50000))
            password = generateServerPassword()
            cred = password
            print('[+] creating account for %s with password %s' % (username,password))
            account_created = self.Create_Account_Selenium_Task(username, password)
            print("\nAccount Created :\n", account_created)
            # os.system('service tor restart')
            if(account_created):
                i = i+1
            else:
                print('[-] name not recorded due to captcha issue')
            time.sleep(2)
    
    async def processRow(self, username, row):
        try:
            messageId = row["id"]
            insertedRecord = await self.db_engine.search_for_record('sociality', False, messageId)
            if(not insertedRecord):
                tweet = row["tweet"]
                print("\n================================================")
                print("\nChecking Tweet",tweet,"\n")
                print("\n================================================")
                result = row.to_json(orient="columns")
                parsed = json.loads(result)
                row_data = json.dumps(parsed, indent=4) 
                record_To_Insert = {
                    "username": username,
                    "messageId": messageId,
                    "text" : tweet,
                    "source": "twitter",
                    "dataType": "submission",
                    "data": row_data,
                    "dateTime": row['date'],
                }
                record_To_Insert["model_results"] ={}
                record_To_Insert["entities_extracted"] ={}
                await self.db_engine.insert_social_record(record_To_Insert)
        except Exception as e:
            print("\nAn exception occurred when running NLP model on tweets")
            print(e)

    async def Search_For_User(self, username, since='2018-01-01', until=todays_date_formatted, verbose = False ):
        try:
            b = twint.Config()
            b.Username = username
            b.Pandas = True
            # b.Limit = 90
            b.Hide_output = True
            b.Since = since
            b.until= until
            twint.run.Search(b)
            Identity_Tweets_List = twint.storage.panda.Tweets_df
            Identity_Data = {"identity": b.Username, "tweets": Identity_Tweets_List}
            Identity_Tweets_List = Identity_Tweets_List.reset_index()
            today = date.today()
            todays_date_formatted = today.strftime('%Y-%m-%d %H:%M:%S')
            for index, row in Identity_Tweets_List.iterrows():
                try:
                    await self.processRow(username, row)
                except Exception as e:
                    print("\nAn exception occurred when pulling tweets")
                    print(e)
        except Exception as e:
                print("\nAn exception occurred when pulling tweets")
                print(e)
                
    async def Search_For_Search_Term(self, searchTerm, no_analysis_flag=True, since='2018-01-01', until=todays_date_formatted, verbose = False ):
        try:        
            b = twint.Config()
            b.Search = searchTerm
            b.Pandas = True
            b.Limit = 90
            b.Hide_output = True
            b.Since = since
            b.until= until
            twint.run.Search(b)
            Identity_Tweets_List = twint.storage.panda.Tweets_df
            Identity_Tweets_List = Identity_Tweets_List.reset_index()  # make sure indexes pair with number of rows
            today = date.today()
            todays_date_formatted = today.strftime('%Y-%m-%d %H:%M:%S')
            t3 = False
            for index, row in Identity_Tweets_List.iterrows():
                author = row["username"]
                print(author)
                try:
                    t3 = Process(target=asyncio.run(self.Search_For_User(author)))
                    t3.start()
                    # t3.join()
                except Exception as e:
                    print(e)
            if(t3):
                t3.join()
        except Exception as e:
            print("\nAn exception occurred when pulling tweets")
            print(e)

social_sites = [Reddit, Twitter]