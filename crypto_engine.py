# ===============================================================================================================
#                                   Check & Install Dependencies
# ===============================================================================================================
#
import random
import time
import re
import string
import secrets
import os
import urllib.request
import json
import time
import csv
import logging, sys, getopt
import array
from datetime import date, datetime
from multiprocessing import Process
import asyncio
try:
    from dotenv import load_dotenv
    load_dotenv('./.env')
except Exception as e:
    print("\ninstalling python-dotenv")
    print(os.system("pip install python-dotenv"))
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
except Exception as e:
    print("\ninstalling multiprocessing")
    print(os.system("pip install multiprocessing"))

class Crypto_Engine():
    def __init__(self, key_dir):
        self.key_dir = key_dir
        if(not os.path.exists(key_dir)):
            os.mkdir(key_dir)
            
    async def createPrivateKey(self, passphrase, save_file_path=False):
        keys_directory = self.key_dir
        key = RSA.generate(2048)
        encrypted_key = key.export_key(passphrase=passphrase, pkcs=8,
                                      protection="scryptAndAES128-CBC")
        if(save_file_path):
             # Save file to path
            file_out = open(save_file_path, "wb")
            file_out.write(encrypted_key)
            file_out.close()
        return encrypted_key

    async def pullPublicKey(self, priv_key_path, passphrase, save_file_path=False ):
        # Read Public key from tsa key 
        encoded_key = open(priv_key_Path, "rb").read()
        key = RSA.import_key(encoded_key, passphrase=passphrase)
        pub_key = key.publickey().export_key()
        if(save_file_path):
            file_out = open(save_file_path, "wb")
            file_out.write(pub_key)
            file_out.close()
        return pub_key

    async def pullPrivateKey(self, priv_key_path, passphrase, save_file_path=False ):
        # Read Private key from tsa key 
        encoded_key = open(priv_key_path, "rb").read()
        key = RSA.import_key(encoded_key, passphrase=passphrase)
        priv_key = key.export_key()
        if(save_file_path):
            file_out = open(save_file_path, "wb")
            file_out.write(priv_key)
            file_out.close()
        return priv_key

    async def encryptData(self, pub_key, text_to_encrypt):
        data = text_to_encrypt.encode("utf-8")
        recipient_key = RSA.import_key(pub_key)
        session_key = get_random_bytes(16)
        # Encrypt the session key with the public RSA key
        cipher_rsa = PKCS1_OAEP.new(recipient_key)
        enc_session_key = cipher_rsa.encrypt(session_key)
        # Encrypt the data with the AES session key
        cipher_aes = AES.new(session_key, AES.MODE_EAX)
        ciphertext, tag = cipher_aes.encrypt_and_digest(data)
        responseData =  {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'), 
            "tag": base64.b64encode(tag).decode('utf-8'), 
            "nonce": base64.b64encode(cipher_aes.nonce).decode('utf-8'), 
            "enc_session_key": base64.b64encode(enc_session_key).decode('utf-8')
        }
        return responseData

    async def decryptData(self, priv_key, passphrase, text_to_decrypt):
        private_key = RSA.import_key(priv_key, passphrase=passphrase)
        enc_session_key = base64.b64decode(text_to_decrypt["enc_session_key"])
        nonce = base64.b64decode(text_to_decrypt["nonce"])
        tag = base64.b64decode(text_to_decrypt["tag"])
        ciphertext = base64.b64decode(text_to_decrypt["ciphertext"])
        # Decrypt the session key with the private RSA key
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        # Decrypt the data with the AES session key
        cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
        data = cipher_aes.decrypt_and_verify(ciphertext, tag)
        return data.decode("utf-8")

    async def encrypt_old_database(self, pub_key, main_file = './tinydb-crypt.json', old_file= './tinydb.json', table_name="analysis_results"):
        db = TinyDB(old_file)
        db2 = TinyDB(main_file)
        db1_table = db.table(table_name)
        db2_table = db2.table(table_name)
        skipped_record_count = 0
        for row in db1_table.all()[::-1]:
            recordToSearch = Query()
            insertedRecord = db2_table.search((recordToSearch.messageId == row["messageId"]))
            if(not insertedRecord):
                print("\n\n\nInserting: ",row["username"],"\n",row["text"],"\n\n")
                tmp_row = row
                tmp_row["username"] = self.encryptData(pub_key, tmp_row["username"])
                tmp_row["text"] = self.encryptData(pub_key, tmp_row["text"])
                tmp_row["source"] = self.encryptData(pub_key, tmp_row["source"])
                tmp_row["dataType"] = self.encryptData(pub_key, tmp_row["dataType"])
                # tmp_row["messageId"] = self.encryptData(pub_key, tmp_row["messageId"])
                # tmp_row["entities_extracted"] = self.encryptData(pub_key, tmp_row["entities_extracted"])
                tmp_row["data"] = self.encryptData(pub_key, tmp_row["data"])
                db2_table.insert(dict(tmp_row))
            else:
                skipped_record_count+=1
