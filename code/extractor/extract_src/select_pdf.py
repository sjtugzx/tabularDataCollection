import os
import json
import random
from shutil import copyfile

files = os.listdir("../marks_main_exp3")

counter = 0

random.shuffle(files)

for file in files:
    if int(file.split("_")[0]) == 1:
        old_pdf_path = os.path.join("../raw_pdf_exp3", file[2: -4])
        new_pdf_path = os.path.join("../select_pdf_exp3", file[2: -4])
        copyfile(old_pdf_path, new_pdf_path)
        counter += 1
    # if counter == 100:
    #     break
    # with open(os.path.join("../marks", file), "r") as f:
    #     data = json.load(f)
    # yes_table += sum(data)
    # no_table += len(data) - sum(data)

# print(yes_pdf, no_pdf, yes_table, no_table)
