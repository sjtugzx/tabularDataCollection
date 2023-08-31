import xlrd
import string
import os

import re

# from tqdm import tqdm
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# import numpy as np
#
# from sklearn.metrics import precision_score


def read_data():
    label_path = "../multi_label.xls"
    label_table = xlrd.open_workbook(label_path).sheets()[0]

    degree_correct = 0
    encode_correct = 0
    format_correct = 0
    mixed_correct = 0

    counter = 0

    for label_rid in range(label_table.nrows):
        if label_rid == 0:
            continue

        pdf_id = str(int(label_table.cell_value(label_rid, 0))).zfill(4)
        available = 1 if int(0 if not str(label_table.cell_value(label_rid, 1)).strip()
                             else label_table.cell_value(label_rid, 1)) != -100 else 0
        degree_label = int(0 if not str(label_table.cell_value(label_rid, 2)).strip()
                           else label_table.cell_value(label_rid, 2))
        encode_label = int(0 if not str(label_table.cell_value(label_rid, 3)).strip()
                           else label_table.cell_value(label_rid, 3))
        format_label = (0 if not str(label_table.cell_value(label_rid, 4)).strip()
                        else label_table.cell_value(label_rid, 4))
        mixed_label = (0 if not str(label_table.cell_value(label_rid, 5)).strip()
                       else label_table.cell_value(label_rid, 5))
        total_label = (0 if not str(label_table.cell_value(label_rid, 7)).strip()
                       else label_table.cell_value(label_rid, 7))
        if available == 0:
            continue

        counter += 1

        excel_path = "../output/excel/{}_0_0_raw.xls".format(pdf_id)

        table = xlrd.open_workbook(excel_path).sheets()[0]
        col_num = table.ncols
        row_num = table.nrows
        cells = []
        for rid in range(row_num):
            for cid in range(col_num):
                cells.append(str(table.cell_value(rid, cid)))

        format_correct += 1 if (check_format(cells, col_num, row_num) == format_label) else 0
        # if not (check_format(cells, col_num, row_num) == format_label):
        #     print(pdf_id, format_label)

        encode_correct += 1 if (check_encode(cells) == encode_label or check_format(cells, col_num, row_num) == 1) else 0
        # if not (check_encode(cells) == encode_label or format_label == 1):
        #     print(pdf_id, encode_label)
        mixed_correct += 1 if (check_mixed_lines(cells) == mixed_label or check_format(cells, col_num, row_num) == 1 or check_encode(cells) == 1) else 0

        if not (check_mixed_lines(cells) == mixed_label or check_format(cells, col_num, row_num) == 1 or check_encode(cells) == 1):
            print(pdf_id, mixed_label)
            check_mixed_lines(cells, True)

    print(format_correct / counter)
    print(encode_correct / counter)
    print(mixed_correct / counter)


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


def check_false_text(cells):
    false_text = re.compile(r"(I．|O．|o．|．I|．O|．o|[0-9IoO]土[0-9IoO]|[0-9IoO]士[0-9IoO])")
    counter = 0
    no_digit_cell = 0
    for cell in cells:
        if cell.strip():
            counter += 1
            if false_text.search(cell):
                return 1
            flag = False
            for char in cell:
                if not char.strip():
                    continue
                if char.isdigit():
                    flag = True
            if not flag:
                no_digit_cell += 1
    if no_digit_cell / max(1, counter) > 0.9:
        return 1
    return 0


# def check_degree(cells):
#     re_degree = re.compile(r"[1-3]?[0-9]?[0-9]°[0-6]?[0-9]′[0-6]?[0-9]″\.?[0-9]?[0-9]?[0-9]?")


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


if __name__ == '__main__':
    # print(longestDupSubstring("14,154,000InfeasibleInfeasible20,212,000"))
    # print(check_mixed_lines(["127 ± 4 Ma130 ± 4 Ma97.7 ± 1.5 Ma"]))
    read_data()
