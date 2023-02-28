# Friendly reminder...
# This is for testing purposes ONLY
# The developers (Dylan.. lol) accept no responsibility for any misuse YOU may cause with this.
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
from collections import Counter
from datetime import date, datetime
from multiprocessing import Process
import asyncio
from social_engine import social_sites
from data_generator import generateServerPassword, generateUsername
import numpy as np
try:
    from PIL import Image, ImageDraw
except Exception as e:
    print("\ninstalling pillow")
    print(os.system("pip install --upgrade protobuf==3.20.0"))
    print(os.system("pip install pillow"))
try:
    from scapy.all import rdpcap
    import pyshark
    import networkx as nx
    import matplotlib.pyplot as plt
except Exception as e:
    print(os.system("pip install networkx[all]"))
    print(os.system("pip install matplotlib"))
    print(os.system("pip install --pre scapy[basic]"))
    print(os.system("pip install pyshark"))
    import networkx as nx
    import matplotlib.pyplot as plt
    import pyshark
    from scapy.all import rdpcap

try:
    from prettytable import PrettyTable
except Exception as e:
    #print(os.system("pip install --user prettytable"))
    from prettytable import PrettyTable
async def concat_images(img1, img2):
    dst = Image.new('RGB', (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst
async def generateImage(filename,values, colors, width, height):
    img  = Image.new( mode = "RGB", size = (width, height) )
    draw = ImageDraw.Draw(img)
    numFeatures = len(values)
    for i in range(0,numFeatures):
        if(i > len(values)):
            x1=(width/numFeatures) * i
            x2=((width/numFeatures) * i) + ((width/numFeatures) * 0)
            draw.rectangle((x1, 0, x2, height), fill=colors[i], outline=(255, 255, 255))
        else:
            x1=(width/numFeatures) * i
            x2=((width/numFeatures) * i) + ((width/numFeatures) * values[i])
            draw.rectangle((x1, 0, x2, height), fill=colors[i], outline=(255, 255, 255))
    print("saving to {0}".format(filename))
    img.save(filename)
    return img
async def genColors(values):
    colors=[]
    # for i in range(0,len(values)):
    for i in range(0,65535):
        r =random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = (r,g,b)
        colors.append(rgb)
    return colors
async def gen_image_from_packet(colors, packet):
    try:
        json_packet = packet # json.dumps(packet)
        today = date.today()
        base_path = os.path.expanduser(os.path.join('./PCAP_IMAGES/'))
        base_path_exist = os.path.exists(base_path)
        if(not base_path_exist):
            os.makedirs(base_path)
        foundOne = False
        selected_pcap_path = False
        quantity_index_ = 0
        while not foundOne:
            todays_date_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            tmp_pcap_path = "{2}PCAP_IMAGE_{0}-{1}.jpeg".format(todays_date_formatted, random.randint(5,55555),base_path)
            pcap_full_path = os.path.join(base_path,tmp_pcap_path)
            pcap_full_path_exists = os.path.exists(pcap_full_path)
            if(not pcap_full_path_exists):
                foundOne = True 
                selected_pcap_path = tmp_pcap_path
            quantity_index_ += 1
        try:
            print(json_packet)
            image = await generateImage(selected_pcap_path, json_packet, colors, 500, 20)
            return image
        except Exception as e:
            print("bunga")
            print(e)
    except Exception as e:
        print(e)
    
    # img = img.resize((36,36),Image.ANTIALIAS)l
    # img.save(pcap_36_path+fil[0:-6]+".bmp")
    
async def gen_images_for_packet_array(packet_list):
    final_image_array = []
    final_image = None
    colors_chosen = await genColors(65535)

    for packet in packet_list:
        tmp_packet = packet.get_raw_packet()

        tmp_img = await gen_image_from_packet(colors_chosen,tmp_packet)
        if(not final_image):
            final_image = tmp_img
        else:
            try:
                final_image = await concat_images(final_image, tmp_img)
            except Exception as e:
                print(e)
    if(final_image):
        # print(final_image)
        final_image.save("./final.jpeg")
        return final_image


class Graphing_Engine():
    # https://github.com/networkx/networkx
    # https://github.com/graphistry/pygraphistry
    def __init__(self, graph_type, graph_data, graph_filepath=False):
        self.graph_type = graph_type or "social"
        self.graph_data = graph_data or []
        self.graph_filepath = graph_filepath
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
                ],
                "phone": [
                    "curl -sSL https://raw.githubusercontent.com/sundowndev/phoneinfoga/master/support/scripts/install | bash ",
                    "https://github.com/AzizKpln/Moriarty-Project.git",
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
            },
            "graph":{
                "misc": [
                    "pip install networkx[all]",
                    "pip install prettytable"
                    "https://github.com/graphiy/pygraphistry",
                ],
                "pcap": [
                    "pip install pyshark"
                    "apt-get install tshark -y"
                ],
                "dashboards": [
                    "https://github.com/graphistry/graph-app-kit.git"
                ]
            }
        }
    async def install(self):
        try:
            base_path = os.path.expanduser(os.path.join('./tools'))
            base_path_exist = os.path.exists(base_path)
            if(not base_path_exist):
                os.makedirs(base_path)
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
                                #print("Error downloading {0}".format(repo))
                                print(e)
                        elif(repo.split(" ")[0] == "curl"):
                            try:
                                p = Process(target=os.system("cd {0} && {1}".format(engine_tool_subcategory_path, repo)))
                                p.start()
                            except Exception as e:
                                #print("Error downloading {0}".format(repo))
                                print(e)
                        elif(repo.split(" ")[0] == "apt-get"):
                            try:
                                p = Process(target=os.system("sudo {1}".format(repo)))
                                p.start()
                            except Exception as e:
                                #print("Error downloading {0}".format(repo))
                                print(e)
                        else:
                            try:
                                p = Process(target=os.system("cd {0} && git clone {1}".format(engine_tool_subcategory_path,repo)))
                                p.start()
                            except Exception as e:
                                #print("Error downloading {0}".format(repo))
                                print(e)

        except Exception as e:
            print(e)

    
    def graph_network_traffic(self, packet_count=1000,interface="any", save_pcap=True):
        try:
            import pyshark
            G = nx.Graph()
            connections = set()
            nodes = set()
            pcap_path_formatted = False
            if(save_pcap):
                pcap_path_formatted = save_pcap
            else:
                today = date.today()
                todays_date_formatted = today.strftime("%Y-%m-%d")
                pcap_path = "{0}_PCAP.pcap".format(todays_date_formatted)
                base_path = os.path.expanduser(os.path.join('./PCAPS'))
                base_path_exist = os.path.exists(base_path)
                if(not base_path_exist):
                    os.makedirs(base_path)
                pcap_full_path = os.path.join(base_path,pcap_path)
                pcap_full_path_exists = os.path.exists(pcap_full_path)
                if(not pcap_full_path_exists):
                    pcap_path_formatted = pcap_full_path
                    foundOne = True 
                else:
                    foundOne = False
                    quantity_index_ = 0
                    while not foundOne:
                        todays_date_formatted = today.strftime("%Y-%m-%d-%H-%M-%S")
                        tmp_pcap_path = "{0}_PCAP{1}.pcap".format(todays_date_formatted, quantity_index_)
                        pcap_full_path = os.path.join(base_path,tmp_pcap_path)
                        pcap_full_path_exists = os.path.exists(pcap_full_path)
                        if(not pcap_full_path_exists):
                            pcap_path_formatted = pcap_full_path
                            foundOne = True 
                        quantity_index_ += 1

            if(pcap_path_formatted):
                srcIps = []
                destIps = []
                capture = pyshark.LiveCapture(interface=interface, include_raw=True, use_json=True,output_file=pcap_path_formatted)
                print("Sniffing next {0} packets...\n".format(packet_count))
                cnt1 = Counter()
                cnt2 = Counter()
                packets = []
                packet_images = []
                
                for packet in capture.sniff_continuously(packet_count=packet_count):
                    try:
                        if 'IP' in packet:
                            # #print(packet)
                            packets.append(packet)
                            source_ip = packet.ip.src
                            dest_ip = packet.ip.dst
                            # #print("just arrived", packet)
                            # #print("\n",source_ip, dest_ip)
                            
                            if(source_ip != "127.0.0.1" or dest_ip != "127.0.0.1"):
                                try:
                                    print('\n\nJust arrived\nSource: {0} -> {1}\n'.format(source_ip, dest_ip))
                                    
                                    nodes.add(source_ip)
                                    nodes.add(dest_ip)
                                    connections.add((source_ip, dest_ip))
                                    cnt1[source_ip] += 1
                                    cnt2[dest_ip] += 1
                                    if(source_ip not in srcIps):
                                        srcIps.append(source_ip)
                                    if(dest_ip not in destIps):
                                        destIps.append(dest_ip)
                                except Exception as e:
                                    print(e)

                                    #print("Exception when trying to add packet src & Destination nodes {0}\n\n".format(e))
                    except Exception as e:
                        print(e)
                            #print("Exception when trying to add packet\n",e)
            
            
                #print("Saving to: {0}".format(pcap_path_formatted))
                src_table = PrettyTable(["IP", "COUNT"])
                dest_table = PrettyTable(["IP", "COUNT"])
                for ip, count in cnt1.items():
                    src_table.add_row([ip, count])
                for ip, count in cnt2.items():
                    dest_table.add_row([ip, count])
                print(src_table.get_string(title="Source IP Address"),"\n")
                print(dest_table.get_string(title="Destination IP Address"),"\n")
                G.add_nodes_from(nodes)
                G.add_edges_from(connections)
                plt.rcParams['savefig.format'] = "jpeg"
                # plt.rcParams['figure.figsize'] = 150, 120

                pos = nx.spring_layout(G, scale=1.0, iterations=100)
                nx.draw(G, pos, node_color='c',edge_color='k', with_labels=True)
                
                asyncio.run(gen_images_for_packet_array(packets))

        except Exception as e:
            print(e)
            return self.graph_network_traffic(packet_count=packet_count,interface="eth0", save_pcap=False)
    
    def graph_pcap(self, pcap_file_path):
        import pyshark
        G = nx.Graph()
        p = False
        connections = set()
        nodes = set()
        srcIps = []
        destIps = []
        cnt1 = Counter()
        cnt2 = Counter()
        read_packets = pyshark.FileCapture(pcap_file_path)
        for packet in read_packets:
             if 'IP' in packet:
                try:
                    source_ip = packet.ip.src
                    dest_ip = packet.ip.dst
                    if(source_ip != "127.0.0.1" or dest_ip != "127.0.0.1"):
                        #print("Adding:\n\tSource: {0}\n\tDest: {1}".format(source_ip,dest_ip))
                        nodes.add(source_ip)
                        nodes.add(dest_ip)
                        connections.add((source_ip, dest_ip))
                        cnt1[source_ip] += 1
                        cnt2[dest_ip] += 1
                        if(source_ip not in srcIps):
                            srcIps.append(source_ip)
                        if(dest_ip not in destIps):
                            destIps.append(dest_ip)
                except Exception as e:
                    #print("Exception when trying to add packet src & Destination nodes")
                    print(e)
        #print("\n\nGenerating image...")
        G.add_nodes_from(nodes)
        G.add_edges_from(connections)
        # plt.rcParams['figure.figsize'] = 1000, 1000
        plt.rcParams['savefig.format'] = "jpeg"
        pos = nx.spring_layout(G, scale=1.0, iterations=100)
        nx.draw(G, pos, node_color='c',edge_color='k', with_labels=True)
        src_table = PrettyTable(["IP", "COUNT"])
        dest_table = PrettyTable(["IP", "COUNT"])
        for ip, count in cnt1.items():
            src_table.add_row([ip, count])
        for ip, count in cnt2.items():
            dest_table.add_row([ip, count])
        #print(src_table.get_string(title="Source IP Address"),"\n")
        file1 = open(str(pcap_file_path).replace(".pcap","_TEXT"), "a")  # append mode
        file1.write(src_table.get_string(title="Source IP Address"))
        #print(dest_table.get_string(title="Destination IP Address"),"\n")
        file1.write(dest_table.get_string(title="Destination IP Address"))
        file1.close()
        return plt.savefig(str(pcap_file_path).replace(".pcap",".jpeg"))
            # return plt

    def graph_social_interactions(self, username, social_engine):
        # Get social interactions
        # Build network with social interactions with target in the center
        # Build different graphs per model metric
        return True
    