import sys
#sys.path.insert(0, 'C:\\Users\\astro_000\\git\\RESEARCH_WordModel\\BNC')
#import bncWork

import nltk
import re
import os
import hashlib
import time
import operator
import math
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.cluster.api import ClusterI
from random import randint
from shutil import copyfile

file_dir = "C:\\Users\\astro\\Desktop\\School\\CSCI249\\Project\\instacart_2017_05_01\\"
aisles = None
#aisle_id, aisle

departments = None
#department_id, department

products = None
#product_id, product_name, aisle_id, department_id

orders = None
#order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order

order_products__train = None
#order_id, product_id, add_to_cart_order, reordered

order_products__prior = None
#order_id, product_id, add_to_cart_order, reordered

user_info = None
#user_id, orders_sum, orders, days_since_prior_std, eval_list,
#   dow_list, hour_list, avg_basket, std_basket, basket_list, avg_reorder,
#   std_reorder, reorder_list, num_unique_items



user_dict = {}

def main():

    global aisles
    global departments
    global products
    global orders
    global order_products__train
    global order_products__prior
    global user_info
    
    aisles = df = pd.read_csv(file_dir + "aisles.csv")
    print("AISLES DONE")
    
    departments = pd.read_csv(file_dir + "departments.csv")
    print("DEPARTMENT DONE")
    
    products = pd.read_csv(file_dir + "products.csv")
    print("PRODUCTS DONE")
    
    orders = pd.read_csv(file_dir + "orders.csv")
    print("ORDERS DONE")

    order_products__train = pd.read_csv(file_dir + "order_products__train.csv")
    print("order_products__train DONE")

    order_products__prior = pd.read_csv(file_dir + "order_products__prior.csv")
    print("order_products__prior DONE")

    user_info = pd.read_csv(file_dir + "user_info.csv")
    print("user_info DONE")

def clean_user_info():
    mark = 9
    rfile = open(file_dir + "user_info.csv", "r")
    wfile = open(file_dir + "user_info_new_" + str(mark - 1) + ".csv", "w")
    lines = rfile.readlines()
    global order_products__train
    global order_products__prior

    #182214 - 183536

    wfile.write(lines[0].split("\n")[0] + ",avg_basket,std_basket,basket_list,"+
                "avg_reorder,std_reorder,reorder_list,num_unique_items\n")
    num_users = len(lines) - 1

    offset = 11500

    base = 40133 + 1
    begin = (base * 4) + (offset * (mark - 4))
    end = begin + offset

    cur_user = 0

    if(end > (num_users + 1)):
        end = num_users + 1

    begin = 40132
    end = 40134
        

    for line in lines[begin:end]:
        cur_user = cur_user + 1
        print((cur_user / (end - begin)) * 100)
        
        u = (line.split("\n")[0]).split(",")
        order_ids = u[2]
        eval_ids = u[5]
        
        order_ids = ((order_ids.split("[")[1]).split("]")[0]).split(":")
        eval_ids  = ((eval_ids.split("[")[1]).split("]")[0]).split(":")

        num_baskets = len(order_ids)
        total_basket_size = 0
        total_reordered = 0
        total_unique = 0

        size_list = []
        reordered_list = []

        for i in range(len(order_ids)):
            o_id = int(order_ids[i])
            
            e = eval_ids[i]
            df = order_products__train
            if(e == '\'prior\''):
                df = order_products__prior

            order = df[df['order_id'] == o_id]
            #print(order)

            basket_size = len(order['reordered'])
            num_reordered = np.sum(order['reordered'])

            size_list.append(basket_size)
            reordered_list.append(num_reordered)

            total_basket_size = total_basket_size + basket_size
            total_reordered = total_reordered + num_reordered

            num_unique = total_basket_size - total_reordered
            total_unique = total_unique + num_unique

            #print(basket_size, num_reordered)

        avg_basket = total_basket_size / num_baskets
        avg_reordered = total_reordered / num_baskets
        std_basket = np.std(size_list)
        std_reordered = np.std(reordered_list)

        str_reordered_list = str(reordered_list)
        str_reordered_list = re.sub(r', ', ":", str_reordered_list)

        str_basket_list = str(size_list)
        str_basket_list = re.sub(r', ', ":", str_basket_list)

        #print(avg_basket, avg_reordered, std_basket, std_reordered)
        
        wfile.write(line.split("\n")[0] + "," + str(avg_basket) + "," +
        str(std_basket) + "," + str(str_basket_list) + "," + str(avg_reordered) + "," +
        str(std_reordered) + "," + str(str_reordered_list) + "," + str(total_unique) + "\n")


    wfile.close()
    rfile.close()

def prepare_user_dict():
    global user_dict
    num_users = 206209

    wfile = open(file_dir + "user_info.csv", "w")
    wfile.write("user_id,orders_sum,orders,sum_reorder,sum_basket_size,"+
                "sum_unique_item_per_basket,days_since_prior_avg,days_since_prior_std,"+
                "eval_list,dow_list,hour_list\n")
    
    #order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
    for user_id in range(1, 206209 + 1):
        print((user_id / num_users) * 100)
        u = orders[orders['user_id'] == user_id]

        u_list = u.values.T.tolist()
        orders_sum = len(u_list[0])
        days_since_prior_avg = np.average(u['days_since_prior_order'])
        days_since_prior_std = np.std(u['days_since_prior_order'])
        eval_list = u_list[2]
        order_list = u_list[0]
        dow_list = u_list[4]
        hour_list = u_list[5]

        v = [orders_sum, order_list, -1, -1, -1, days_since_prior_avg, days_since_prior_std,
                              eval_list, dow_list, hour_list]
        wfile.write(str(user_id) + ":" + str(v[0]) + ":" + str(v[1]) + ":" +
                    str(v[2]) + ":" + str(v[3]) + ":" + str(v[4]) +
                    str(v[5]) + ":" + str(v[6]) + ":" + str(v[7]) + ":" +
                    str(v[8]) + "," + str(v[9]) + "\n")


    #print("NOW MAKING FILE:")
    #wfile = open(file_dir + "user_info.csv", "w")

    #for k in user_dict:
    #    v = user_dict[k]
    #    wfile.write(str(k) + "," + str(v[0]) + "," + str(v[1]) + "," +
    #                str(v[2]) + "," + str(v[3]) + "," + str(v[4]) +
    #                str(v[5]) + "," + str(v[6]) + "," + str(v[7]) + "\n")
    wfile.close()

def get_file_dict(fileName):
    rfile = open(file_dir + fileName, "r", encoding="utf-8")
    lines = rfile.readlines()
    aisles = {}

    for line in lines[1:]:
        split = (line.split("\n")[0]).split(",")

        key = split[0]
        values = []

        for x in range(1, len(split)):
            values.append(split[x])

        aisles[int(key)] = values

    rfile.close()
    return aisles

main()
