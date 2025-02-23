import json
import csv

def load_L1_json():
    with open('L1.json') as json_file:
        l1_data = json.load(json_file)
    return l1_data

def load_Q1_csv():
    q1_csv_data = []
    with open('Q1.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            q1_csv_data.append((float(row[0]), float(row[1]))) 
    return q1_csv_data

def load_L1_csv():
    l1_csv_data = []
    with open('L1.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            l1_csv_data.append((float(row[0]), float(row[1]))) 
    return l1_csv_data

def load_Q2_csv():
    q2_csv_data = []
    with open('Q2.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            q2_csv_data.append((float(row[0]), float(row[1]), float(row[2]))) 
    return q2_csv_data

