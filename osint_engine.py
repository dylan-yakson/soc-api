import random
import time
import re
import string
import secrets
import os, stat
import urllib.request
import json
import time
import csv
import shutil
import logging, sys, getopt
import array
from datetime import date, datetime
from multiprocessing import Process
import asyncio
from social_engine import social_sites
try:
    from tools.custom.LicensePlateDetector.detect import run_plate_detection_model
    from tools.custom.email2phone import set_proxy_list, start_scrapping
    from tools.custom.iKy_functions import search_username_iky, pretty_print
except Exception as e:
    tmp_dir = './tools/tmp'
    custom_dir = './tools/custom/'
    custom_zip_dir = './custom.zip'
    tmp_custom_dir = './tools/tmp/custom/'
    today = date.today() 
    # todays_date_formatted = today.strftime("%Y-%m-%d-%H-%M-%S")
    # custom_archive_zip_dir = "custom{0}".format(todays_date_formatted)
    custom_archive_zip_dir = "custom"

    shutil.make_archive(custom_archive_zip_dir, 'zip', custom_dir)

    try:
        def remove_readonly(func, path, _):
            "Clear the readonly bit and reattempt the removal"
            os.chmod(path, stat.S_IWRITE)
            func(path)
        base_path = os.path.expanduser(os.path.join('./tools'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        else:
            shutil.rmtree(tmp_dir, onerror=remove_readonly)
        import zipfile
        with zipfile.ZipFile(custom_zip_dir, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            # shutil.move(tmp_custom_dir, custom_dir)
    except Exception as e:
        print(e)
    from tools.custom.LicensePlateDetector.detect import run_plate_detection_model
    from tools.custom.email2phone import set_proxy_list, start_scrapping
    from tools.custom.iKy_functions import search_username_iky, pretty_print

try:
    import cv2
except Exception as e:
    print(os.system("pip install opencv-python==4.3.0.36"))
    import cv2
try:
    import tesserocr
except Exception as e:
    print(os.system("pip install tesserocr"))
    import tesserocr

class Osint_Engine():
    def __init__(self, social_sites=[], shodan_api_key="" ):
        self.shodan_api_key = shodan_api_key
        self.social_sites = social_sites
        self.toolset = {
            "osint": {
                "username": [
                    "https://github.com/sherlock-project/sherlock.git",
                    "https://github.com/soxoj/maigret.git",
                ],
                "email": [
                    "pip install holehe",
                    "https://github.com/mxrch/GHunt",
                    "https://github.com/khast3x/h8mail",
                    "https://github.com/kennbroorg/iKy.git",
                ],
                "phone": [
                    "curl -sSL https://raw.githubusercontent.com/sundowndev/phoneinfoga/master/support/scripts/install | bash ",
                    "https://github.com/AzizKpln/Moriarty-Project.git",
                    "https://github.com/martinvigo/voicemailautomator.git",
                ],
                "website": [
                    "https://github.com/laramies/theHarvester",
                    "https://github.com/six2dez/reconftw",
                    "https://github.com/RustScan/RustScan",
                    "https://github.com/gildas-lormeau/SingleFile",
                ],
                "image": [
                    "pip install deepface",
                    "https://github.com/Greenwolf/social_mapper",
                ],
                "audio": [
                    "https://github.com/Delta-ML/delta",
                ],
                "video": [
                    "https://github.com/omar178/Emotion-recognition",
                ],
                "github": [
                    "https://github.com/eth0izzle/shhgit.git",
                    "https://github.com/BishopFox/GitGot",
                ],
                "deepweb": [
                    "https://github.com/DedSecInside/TorBot",
                ],
                "company": [
                    "https://github.com/l4rm4nd/XingDumper",
                    "https://github.com/j3ssie/metabigor",
                    "https://github.com/nodauf/GoMapEnum.git",
                ],
                "location": [
                    "https://github.com/thewhiteh4t/seeker",
                    "https://github.com/jofpin/trape",
                ],
                "person": [
                    "https://github.com/leebaird/discover",
                    "https://github.com/Lucksi/Mr.Holmes.git",
                    "https://github.com/LandGrey/pydictor",
                    "https://github.com/Mebus/cupp",
                ],
                "misc": [
                    "https://github.com/opsdisk/pagodo",
                    "https://github.com/alephdata/aleph",
                ]
            }
        }
    async def install(self):
        
        try:
           
            for category_name in self.toolset:
                engine_tool_path = os.path.join(base_path,category_name)
                engine_tool_path_exist = os.path.exists(engine_tool_path)
                if(not engine_tool_path_exist):
                    os.makedirs(engine_tool_path)
                for sub_category_name in self.toolset[category_name]:
                    engine_tool_subcategory_path = os.path.join(engine_tool_path,sub_category_name)
                    engine_tool_subcategory_path_exist = os.path.exists(engine_tool_subcategory_path)
                    if(not engine_tool_subcategory_path_exist):
                        os.makedirs(engine_tool_subcategory_path)
                    for repo in self.toolset[category_name][sub_category_name]:
                        if(repo.split(" ")[0] == "pip"):
                            try:
                                p = Process(target=os.system("pip install {1}".format(repo)))
                                p.start()
                            except Exception as e:
                                print("Error downloading {0}".format(repo))
                                print(e)
                        elif(repo.split(" ")[0] == "curl"):
                            try:
                                p = Process(target=os.system("cd {0} && {1}".format(engine_tool_subcategory_path, repo)))
                                p.start()
                            except Exception as e:
                                print("Error downloading {0}".format(repo))
                                print(e)
                        else:
                            try:
                                p = Process(target=os.system("cd {0} && git clone {1}".format(engine_tool_subcategory_path,repo)))
                                p.start()
                            except Exception as e:
                                print("Error downloading {0}".format(repo))
                                print(e)

        except Exception as e:
            print(e)

    async def email_analysis(self, email_to_review):
        #[X] https://github.com/megadose/holehe (Alt accounts via email)
        # https://github.com/martinvigo/email2phonenumber (Phone number from Email)
        # https://github.com/khast3x/h8mail (has it been pwnd?)
        try:
            import holehe
        except Exception as e:
            print("\ninstalling holehe")
            print(os.system("pip install holehe"))
        try:
            import httpx
        except Exception as e:
            print("\ninstalling httpx")
            print(os.system("pip install httpx"))
         
        from holehe.modules.social_media.snapchat import snapchat
        from holehe.core import get_functions, import_submodules
        modules = import_submodules("holehe.modules")
        websites = get_functions(modules,None)
        holehe_out = []
        email_to_phone_out = []

        async def runFunction(email, function, out):
            try:
                client = httpx.AsyncClient()
                await function(email, client, out)
                # print(out)
                await client.aclose()
            except Exception as e:
                print(e)
        # for website in websites:
        #     try:
        #         await runFunction(email_to_review, website, holehe_out)
        #     except Exception as e:
        #         print(e)

        start_scrapping(email_to_review, None)
        return holehe_out

    def domain_analysis(self, domain_to_review, pentest=False):
        print("Reviewing domain: {0}".format(domain_to_review))
        base_path = os.path.expanduser(os.path.join('./tools'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        engine_tool_path = os.path.join(base_path,"osint")
        engine_tool_path_exist = os.path.exists(engine_tool_path)
        if(not engine_tool_path_exist):
            os.makedirs(engine_tool_path)

        engine_tool_subcategory_path = os.path.join(engine_tool_path,"website")
        engine_tool_subcategory_path_exist = os.path.exists(engine_tool_subcategory_path)
        if(not engine_tool_subcategory_path_exist):
            os.makedirs(engine_tool_subcategory_path)

        tool_path = os.path.join(engine_tool_subcategory_path,"reconftw")
        tool_path_exist = os.path.exists(tool_path)
        if(not tool_path_exist):
            os.system("cd {0} && git clone https://github.com/six2dez/reconftw".format(engine_tool_subcategory_path))
            os.system("cd {0} && sudo ./install.sh".format(tool_path))

        # results_path = os.path.join('./', "domain-{0}_results_tmp.txt".format(random.randint(0,100)))
        os.system("cd {0} && sudo ./reconftw.sh -d {1} -r".format(tool_path, domain_to_review))
        os.system("cp -r {0}/Recon/{1} ./".format(tool_path, domain_to_review))
        # domain_results = ""
        # with open(results_path, 'r') as results:
        #     domain_results = results.read()
        # os.remove(results_path) 
        # return domain_results

    async def image_analysis(self, images_to_review=[], fn_type="attributes"):
        try:
            from deepface import DeepFace
        except Exception as e:
            print("\ninstalling DeepFace")
            print(os.system("pip install deepface"))
            from deepface import DeepFace
        # try:
        #     from retinaface import RetinaFace
        # except Exception as e:
        #     print("\ninstalling retinaface")
        #     print(os.system("pip install retinaface"))
        #     from retinaface import RetinaFace
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        metrics = ["cosine", "euclidean", "euclidean_l2"]
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        selected_backend = 4
        selected_model = 1
        def faceVerification(img1_path, img2_path):
            result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = models[selected_model], detector_backend = backends[selected_backend])
            # result = DeepFace.verify(img1_path = img_to_search_path, img2_path = img2_path, distance_metric = metrics[1])
            return result

        def findSimilarFaces(img_to_search_path, db_path="./my_db"):
            df = DeepFace.find(img_path = img_to_search_path, db_path = db_path, model_name = models[selected_model], detector_backend = backends[selected_backend])
            # df = DeepFace.find(img_path = img_to_search_path, db_path = db_path, distance_metric = metrics[1])
            return df 

        def analyzeFace(img_to_search_path):
            obj = DeepFace.analyze(img_path = img_to_search_path, actions = ['age', 'gender', 'race', 'emotion'], model_name = models[selected_model], detector_backend = backends[selected_backend])
            return obj

        def detectFaces(img_to_search_path):
            face = DeepFace.detectFace(img_path = img_to_search_path, target_size = (224, 224), detector_backend = backends[selected_backend])
            result = {
                "retina": resp,
                "deepface": face
            }
            return face
        if(fn_type == "attributes"):
            results = []
            for imagePath in images_to_review:
                analysis = analyzeFace(imagePath)
                results.append(analysis)
            return results
        elif(fn_type == "similar"):
            results = []
            db_path = "./my_db"
            for imagePath in images_to_review:
                analysis = findSimilarFaces(imagePath, db_path)
                results.append(analysis)
            return results
        elif(fn_type == "verify"):
            results = faceVerification(images_to_review[0], images_to_review[1])
            return results
        elif(fn_type == "detectFaces"):
            results = []
            db_path = "./my_db"
            for imagePath in images_to_review:
                analysis = detectFaces(imagePath)
                results.append(analysis)
            return results
        elif(fn_type == "detectLicensePlates"):
            results = []
            db_path = "./my_db"
            for imagePath in images_to_review:
                analysis = detectFaces(imagePath)
                results.append(analysis)
            return results
        else:
            return 'Please provide fn_type: attributes, similar, verify'

    def username_analysis(self, username_to_review, origin):
        try:
            from requests_futures.sessions import FuturesSession
        except Exception as e:
            print("\ninstalling requests_futures")
            print(os.system("pip install requests_futures"))
            from requests_futures.sessions import FuturesSession

        try:
            import requests
        except Exception as e:
            print("\ninstalling requests")
            print(os.system("pip install requests"))
            import requests

        try:
            from torrequest import TorRequest
        except Exception as e:
            print("\ninstalling TorRequest")
            print(os.system("pip install torrequest"))
            from torrequest import TorRequest

        try:
            from colorama import init
        except Exception as e:
            print("\ninstalling colorama")
            print(os.system("pip install colorama"))
            from colorama import init

        # Kinda just... hackily... ripped sherlock apart and stole everything... took like 5 min.. need to be reworked..
        from tools.custom.sherlock.sherlock import checkUsername
        iky_output = search_username_iky(username_to_review)
        sherlock_output = checkUsername(username_to_review)
        output = {
            "iky": iky_output,
            "sherlock": sherlock_output
        }
        return checkUsername(username_to_review)

    def phone_analysis(self, phone_to_review, path_to_dl):
        base_path = os.path.expanduser(os.path.join('./tools'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        engine_tool_path = os.path.join(base_path,"osint")
        engine_tool_path_exist = os.path.exists(engine_tool_path)
        if(not engine_tool_path_exist):
            os.makedirs(engine_tool_path)

        engine_tool_subcategory_path = os.path.join(engine_tool_path,"phone")
        engine_tool_subcategory_path_exist = os.path.exists(engine_tool_subcategory_path)
        if(not engine_tool_subcategory_path_exist):
            os.makedirs(engine_tool_subcategory_path)

        tool_path = os.path.join(engine_tool_subcategory_path,"phoneinfoga")
        tool_path_exist = os.path.exists(tool_path)
        if(not tool_path_exist):
            os.system("cd {0} && curl -sSL https://raw.githubusercontent.com/sundowndev/phoneinfoga/master/support/scripts/install | bash ".format(engine_tool_subcategory_path))
        results = os.system("{0} scan -n {1} > {2}".format(tool_path, phone_to_review, path_to_dl))
        return results

    async def company_analysis(self, company_to_review):
        # Find emails: https://hunter.io/
        # Dorks: https://github.com/opsdisk/pagodo
        # Company Website Screenshot https://github.com/gildas-lormeau/SingleFile
        # https://github.com/m8r0wn/crosslinked ( LinkedIn - Employee names from organization)
        # https://freepeoplesearchtool.com/ ( LinkedIn - Search without Account)
        return 'test'
        
    async def person_analysis(self, fname, lname="", address=False):
        # Spokeo
        # Dorks: https://github.com/opsdisk/pagodo
        # https://freepeoplesearchtool.com/ ( LinkedIn - Search without Account)
        # Find potential usernames from entity extraction
        return 'test'

    async def audio_analysis(self, audio_to_review):
        # https://pytorch.org/audio/main/tutorials/speech_recognition_pipeline_tutorial.html
        return 'test'
        
    async def vides_analysis(self, audio_to_review):
        # https://pytorch.org/audio/main/tutorials/speech_recognition_pipeline_tutorial.html
        # tie into smartphone camera via url in browser (ngrok - seeker)
        return 'test'

    async def github_analysis(self, profile_url_to_review):
        # https://github.com/eth0izzle/shhgit.git
        # https://github.com/BishopFox/GitGot
        return 'test'

    def detect_plates_from_image(self, image_path=b"./test.jpg"):
        found_plates = run_plate_detection_model(image_path)
        print(found_plates)
        return found_plates

    def train_models(self, plates_images_path="./license_detection/license-plates/images"):
        image = train_plate_detection_model(plates_images_path)
        return image

    async def location_analysis(self, record_to_review):
        # Take pictures from google maps
        # Plot nearby businesses
        # Plot building structure using openCV
        # record_to_review:
        # {
        #     "record": "",
        #     "record_type": "address"
        # }
        # or 
        # {
        #     "record": [lat,lng],
        #     "record_type": "coordinates"
        # }
        return 'test'