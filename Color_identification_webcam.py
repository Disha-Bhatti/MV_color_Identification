# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:08:59 2022

@author: PRATHIBHA
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from scipy.spatial import KDTree
from webcolors import (css3_hex_to_names, hex_to_rgb)


vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f' {names[index]}'    


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    return image

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    a=0
    for x in rgb_colors:
        a+=1
        print("Hex to Colour:",hex_colors[a-1],"-->", convert_rgb_to_names(x)) 
        
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.savefig('output/plots%d.jpg' %count)
        plt.show()
    return rgb_colors[0]


 
while success:   
    success,image = vidcap.read()
    cv2.imshow("frame",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (1200, 600))
    plt.imshow(resized_image)
    plt.savefig('input/ploted_rgb%d.jpg' %count)    
    plt.show()
    get_colors(image, 5, True)
    print('Read a new frame: ', success)
    count += 1  

  









