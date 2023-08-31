import xlrd
import string
import os

import re

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

from sklearn.metrics import precision_score


def read_data():
    with open("../500label.txt", "r") as f:
        negs = set([item.strip() for item in f.readlines()])
    dataset = []
    correct = 0
    counter = 0
    total = 0
    total_correct = 0
    for i in tqdm(range(209, 1056)):
        pdf_id = str(i).zfill(4)
        excel_path = "../output/excel/{}_0_0_raw.xls".format(pdf_id)
        if not os.path.exists(excel_path):
            continue
        table = xlrd.open_workbook(excel_path).sheets()[0]
        col_num = table.ncols
        row_num = table.nrows
        cells = []
        for rid in range(row_num):
            for cid in range(col_num):
                cells.append(str(table.cell_value(rid, cid)))



        feature = get_feature_array(col_num, row_num, cells)
        # # false_encode_flag = check_encode(cells)
        # false_format_flag = check_format(cells)
        if str(i) in negs:
            label = 0
        else:
            label = 1

        dataset.append([feature, label, i])


        if not (check_format(cells, col_num, row_num) or check_encode(cells) or check_mixed_lines(cells)):
            if label == 1:
                correct += 1
                total_correct += 1
            else:
                print(i)
            counter += 1
        else:
            if label == 0:
                total_correct += 1
        total += 1

    print(counter)
    print(correct / counter)

    print(total)
    print(total_correct / total)
        # # if label == 1 and false_encode_flag == 1:
        # #     print(i)
        # if false_format_flag == 0:
        #     print(i)
        #     # exit(0)

    print("Positive instance num:", 1000 - len(negs))
    print("Negative instance num:", len(negs))
    return dataset


def check_degree(cells):
    seps = "°\"'‘′’＇″Oo` "
    meas = "WwEeSsNn日F"
    tails = "\+\-\=\:"
    re_degree = re.compile(r"[1-3]?[0-9]?[0-9]°[0-6]?[0-9]′[0-6]?[0-9]″\.?[0-9]?[0-9]?[0-9]?")
    re_degree_raw = re.compile(r"[1-3]?[0-9]?[0-9]([°\"'‘′’＇″¢Oo`]|\(cid:[0-9]+\))[0-6]?[0-9]([°\"'‘′’＇″¢Oo`]|\(cid:[0-9]+\))[0-6]?[0-9]([°\"'‘′’＇″¢Oo`]|\(cid:[0-9]+\))\.?[0-9]?[0-9]?[0-9]?")

    precise = 0
    raw = 0
    for cell in cells:
        if cell.strip():
            if re_degree_raw.search(cell):
                precise += 1
                print(cell)
    print(precise)


def check_encode(cells):
    false_encode = re.compile(r"\(cid:[0-9]+\)")
    for cell in cells:
        if cell.strip():
            if false_encode.search(cell):
                return 1
    return 0


def check_mixed_lines(cells, print_false_cell=False):
    count = 0
    false_count = 0
    punctuation_set = set(string.punctuation)
    for cell in cells:
        if cell.strip():
            # cell = re.findall(r'\(.*?\)', cell)
            count += 1
            char_count, number_count, dot_count, pn_count, minus_count, plus_count, at_count = 0, 0, 0, 0, 0, 0, 0
            for char in cell:
                if not char.strip():
                    continue
                char_count += 1
                if char.isdigit() or char in punctuation_set or char in "Ee±–+":
                    number_count += 1
                if char == ".":
                    dot_count += 1
                if char == "±":
                    pn_count += 1
                if char == "–":
                    minus_count += 1
                # if char == "+":
                #     plus_count += 1
                if char == "@":
                    at_count += 1
            dot_count -= pn_count
            dot_count -= minus_count
            dot_count -= at_count
            dot_count -= len(re.findall(r'\(.*?\)', cell))
            if number_count / max(1, char_count) > 0.7 and (dot_count >= 2 or pn_count >= 2 or minus_count >= 2):
                if print_false_cell:
                    print(cell)
                false_count += 1
            elif number_count / max(1, char_count) > 0.7 and len(longestDupSubstring(cell)) > 4:
                if print_false_cell:
                    print(cell)
                false_count += 1
    return 1 if false_count / max(1, count) > 0 else 0


# def check_degree(cells):
#     re_degree = re.compile(r"[1-3]?[0-9]?[0-9]°[0-6]?[0-9]′[0-6]?[0-9]″\.?[0-9]?[0-9]?[0-9]?")


def longestDupSubstring(s: str) -> str:
    left = 0
    right = 1
    res = ""
    n = len(s)
    while right < n:
        if s[left:right] in s[left + 1:]:
            if right - left > len(res):
                res = s[left:right]
            right += 1
            continue
        left += 1
        if left == right:
            right += 1
    return res

def check_format(cells, col_num, row_num):
    average_space_rate = 0
    words_cell_count = 0
    long_cell_counter = 0
    for cell in cells:
        # if len(cell) > 300:
        #     long_cell_counter += 1
        if cell.strip():
            words_cell_count += 1
            average_space_rate += cell.count(" ")
    average_space_rate /= max(1, len("".join(cells)))
    return 1 if (words_cell_count <= 20 or row_num <= 5 or col_num <= 3 or average_space_rate > 0.4) else 0


def get_feature_array(col_num, row_num, cells):
    feature = [0.0 for _ in range(9)]
    if len(cells) == 0:
        return feature
    feature[0] = col_num / 20
    feature[1] = row_num / 20
    punctuation_set = set(string.punctuation)

    for cell in cells:
        if cell.strip() == "":
            feature[2] += 1
        else:
            for char in cell:
                if char.isdigit():
                    feature[3] += 1
                elif char in set(punctuation_set):
                    feature[4] += 1

            feature[5] += cell.count(" ")
            feature[6] += 1 if cell.find("(cid") > -1 else 0
            feature[7] += 1 if cell.count(".") > 1 else 0
            feature[8] += len(cell)

    for i in range(2, 9):
        feature[i] /= len(cells)

    return feature


def classify(dataset):
    print("Number of instance:", len(dataset))
    print("Number of K:", 5)

    kf = KFold(n_splits=5, shuffle=True)
    X = np.array([item[0] for item in dataset])
    y = np.array([item[1] for item in dataset])

    counter = 1
    acc_sum = 0
    for train_set, test_set in kf.split(dataset):

        X_train = X[train_set]
        y_train = y[train_set]

        X_test = X[test_set]
        y_test = y[test_set]

        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # print(y_test)
        # print(np.array([[item[1], item[2]] for item in test_set])[y_test != y_pred])
        precision = precision_score(y_test, y_pred)
        print(precision)
        acc = sum(y_test == y_pred) / len(y_test)
        # print(y_test[y_test != y_pred])
        print("Acc of fold {}:".format(counter), acc)
        counter += 1
        acc_sum += acc
    print("Average acc:", acc_sum / (counter - 1))


if __name__ == '__main__':
    data = read_data()
    classify(data)
