import json
import csv

with open('L1.json') as json_file:
    l1_data = json.load(json_file)

with open('Q1.json') as json_file:
    q1_data = json.load(json_file)

l1_csv_data = []
with open('L1.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        l1_csv_data.append((float(row[0]), float(row[1]))) 

q2_csv_data = []
with open('Q2.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        q2_csv_data.append((float(row[0]), float(row[1]), float(row[2])))
