# encoding=utf-8
import os
from file_io import *


result_file = "../output/test_results.tsv"
label_file = "../data/s/test.csv"


def get_results_labels(results):
    output = []
    for item in results:
        if float(item[0]) > float(item[1]):
            output.append(0)
        else:
            output.append(1)
    return output


def get_labels(data):
    output = []
    for item in data:
        output.append(int(item[2]))
    return output


def get_results():
    results_data = read_tsv_file(result_file)
    results_labels = get_results_labels(results_data)

    labels_data = read_csv_file(label_file)
    labels = get_labels(labels_data)

    T_TP = 0  
    T_TN = 0  
    T_FP = 0  
    T_FN = 0  

    F_TP = 0  
    F_TN = 0 
    F_FP = 0  
    F_FN = 0  

    wrong_prediction = []
    right_prediction = []
    assert(len(results_labels) == len(labels))
    for i in range(len(labels)):
        if labels[i] == results_labels[i]: 
            right_prediction.append(i)
            if labels[i] == 1:
                T_TN += 1
                F_TP += 1
            else:
                T_TP += 1
                F_TN += 1
        else:  
            wrong_prediction.append(i)
            if labels[i] == 1:
                T_FP += 1
                F_FN += 1
            else:
                T_FN += 1
                F_FP += 1

    try:
        T_acc = (T_TP + T_TN) / (T_TP + T_TN + T_FN + T_FP)
    except:
        T_acc = 0
    try:
        T_prec = T_TP / (T_TP + T_FP)
    except:
        T_prec = 0
    try:
        T_rec = T_TP / (T_TP + T_FN)
    except:
        T_rec = 0
    try:
        T_F1 = (2 * T_prec * T_rec) / (T_prec + T_rec)
    except:
        T_F1 = 0

    try:
        F_acc = (F_TP + F_TN) / (F_TP + F_TN + F_FN + F_FP)
    except:
        F_acc = 0
    try:
        F_prec = F_TP / (F_TP + F_FP)
    except:
        F_prec = 0
    try:
        F_rec = F_TP / (F_TP + F_FN)
    except:
        F_rec = 0
    try:
        F_F1 = (2 * F_prec * F_rec) / (F_prec + F_rec)
    except:
        F_F1 = 0

    return T_acc, T_prec, T_rec, T_F1, F_prec, F_rec, F_F1
