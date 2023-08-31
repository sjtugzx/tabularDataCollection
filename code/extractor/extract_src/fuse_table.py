import os
import json
from shutil import copyfile
import xlrd
import re
import pandas as pd
from tqdm import tqdm
import random

from multiprocessing import Pool

from subsclassify import check_format, check_encode, check_mixed_lines, check_false_text


MARKS_DIR = "../marks"
PDF_DIR = "../pdfs"
EXCEL_DIR = "../output/excel"
OUTPUT_FILE_NAME = "output_final.xls"
META_DATA_DIR = "../meta_data.json"
PROCESS_NUM = 30


def check_value(v):
    if v.strip():
        keyword = re.compile(r"[A-Z]|[a-z]")
        if keyword.search(v):
            return False
        else:
            return True
    return False


def check_sm_nd(cell):
    keyword = re.compile(r"(ε|ɛ|e|E)Nd ?\((\d+|t|T)\)|1?43 ?Nd|1?44 ?Nd|1?47 ?Sm|\(cid:\d+\)Nd|fSm\/Nd|Sm\/Nd")
    if keyword.search(cell):
        return True
    else:
        return False


def check_sample(cell):
    if cell.lower().find("sample") > -1:
        return True
    return False


def get_age(text):
    pattern = re.compile(r"[0-9]+.?[0-9]* ?(± ?[0-9]+.?[0-9]* )?Ma")
    res = pattern.finditer(text)
    res = [item for item in res]
    if len(res) == 0:
        return ""
    if len(res) > 1:
        return res[0].group() + " (?)"
    return res[0].group()


def check_excel(excel_path):
    table = xlrd.open_workbook(excel_path).sheets()[0]
    col_num = table.ncols
    row_num = table.nrows
    cells = []
    for rid in range(row_num):
        for cid in range(col_num):
            cells.append(str(table.cell_value(rid, cid)))

    text_ok = check_false_text(cells)
    format_ok = check_format(cells, col_num, row_num)
    encode_ok = check_encode(cells)
    mixed_lines_ok = check_mixed_lines(cells)
    return format_ok or encode_ok or mixed_lines_ok or text_ok



def matching_keyword(text, keyword_list):
    re_string = ""
    for keyword in keyword_list:
        re_string += r"\b" + keyword + r"\b|"
    re_string = re_string[: -1]
    pattern = re.compile(re_string, re.IGNORECASE)
    res = pattern.finditer(text)
    res = [item for item in res]
    if len(res) == 0:
        return ""
    if len(res) > 1:
        res[0].group() + " (?)"
    return res[0].group()


def get_tect_from_title(title):
    pattern = re.compile(r" (at|in|from|of|on) [(a-zA-Z–\-'äöüÄÖÜß) ]*[A-Z]+[(a-zA-Z–\-'äöüÄÖÜß) ]*,( [(a-zA-Z–\-'äöüÄÖÜß) ]*[A-Z]+[(a-zA-Z–\-'äöüÄÖÜß) ]*,)? [(a-zA-Z–\-'äöüÄÖÜß) ]*[A-Z]+[(a-zA-Z–\-'äöüÄÖÜß) ]*")
    res = pattern.search(title)
    if not res:
        return "", "", ""

    match_text = res.group().strip()
    match_text = match_text[3: ] if match_text[0: 4] != "from" else match_text[5: ]
    sub_tect = ""
    if match_text.count(",") == 1:
        tect, geo_tect = match_text.split(",")
    else:
        sub_tect, tect, geo_tect = match_text.split(",")
    return geo_tect, tect, sub_tect


def get_pluton_from_text(text):
    pattern = re.compile(r"\b([A-Z][(a-zA-Z–\-'äöüÄÖÜß)]* )+pluton\b")
    res = pattern.search(text)
    if not res:
        return ""
    return res.group().strip()[: -7]


def get_raw_tect_from_title(title):
    pattern = re.compile(r" (at|in|from|of|on)( the)?( [a-zA-Z\-'äöüÄÖÜß]*)?( \(?[A-Z][a-zA-Z\-'äöüÄÖÜß]*\)?)+")
    res = pattern.finditer(title)
    res = [item for item in res]
    if len(res) == 0:
        return "", "", ""

    clean_res = []
    for item in res:
        item = item.group().strip()
        item = item[3: ] if item[0: 4] != "from" else item[5: ]
        clean_res.append(item)

    if len(clean_res) >= 3:
        sub_tect, tect, geo_tect = clean_res[-3], clean_res[-2], clean_res[-1]

    if len(clean_res) == 2:
        sub_tect, tect, geo_tect = "", clean_res[-2], clean_res[-1]

    if len(clean_res) == 1:
        sub_tect, tect, geo_tect = "", "", clean_res[-1]
    
    # print(sub_tect, tect, geo_tect)
    
    return geo_tect, tect, sub_tect


def get_header_type(cell):
    # 1. Sample
    if cell.lower().find("samp") > -1:
        return "Sample"

    # 4. Subtectonic unit/Sub groups
    if cell.lower().find("subtectonic") > -1 or cell.lower().find("sub groups") > -1 or cell.lower().find("subgroups") > -1:
        return "Subtectonic unit/Sub groups"

    # 2. Nation/Region/GeoTectonic unit/Groups
    if cell.lower().find("nation") > -1 or cell.lower().find("region") > -1 or cell.lower().find("ocean") > -1 or cell.lower().find("locality") > -1:
        return "Nation/Region/GeoTectonic unit/Groups"

    # 3. GeoTectonic unit/Groups/GeoTectonic unit/Groups
    if cell.lower().find("geotectonic") > -1 or cell.lower().find("groups") > -1:
        return "Nation/Region/GeoTectonic unit/Groups"

    # 5. Tectonic unit
    if cell.lower().find("tectonic") > -1:
        return "Tectonic unit"

    if cell.lower().find("latitude") > -1:
        return "Latitude"

    if cell.lower().find("longitude") > -1:
        return "Longitude"

    # 6. Lithology
    if cell.lower().find("lithology") > -1 or cell.lower().find("rock type") > -1 or cell.lower().find("rock") > -1 or cell.lower().find("description") > -1:
        return "Lithology"

    # 7. Pluton
    if cell.lower().find("pluton") > -1:
        return "Pluton"

    # 8. Age(Ma)
    if cell.lower().find("age") > -1:
        return "Age(Ma)"

    # 9. Sm
    if cell.find("(") > -1:
        current_cell = cell[: cell.find("(")]
    elif cell.find("[") > -1:
        current_cell = cell[: cell.find("[")]
    else:
        current_cell = cell
    current_cell = re.sub('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), "", current_cell)
    if current_cell.lower().strip() == "sm" or current_cell.lower().strip() == "sm ppm":
        return "Sm"

    # # 10. Sr
    # if cell.find("(") > -1:
    #     current_cell = cell[: cell.find("(")]
    # else:
    #     current_cell = cell
    # if current_cell.lower().strip() == "sr":
    #     return "Sr"

    # 11. Nd
    if cell.find("(") > -1:
        current_cell = cell[: cell.find("(")]
    elif cell.find("[") > -1:
        current_cell = cell[: cell.find("[")]
    else:
        current_cell = cell
    current_cell = re.sub('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), "", current_cell)
    if current_cell.lower().strip() == "nd" or current_cell.lower().strip() == "nd ppm":
        return "Nd"

    # 12. 147Sm/144Nd
    keyword = re.compile(r"1?47 ?Sm")
    if keyword.search(cell):
        return "147Sm/144Nd"

    # 13. 143Nd/144Nd
    keyword = re.compile(r"1?43 ?Nd")
    if keyword.search(cell):
        return "143Nd/144Nd"

    # 14. 2σ
    keyword = re.compile(r"2σ|± ?2")
    if "sr" not in cell.lower() and keyword.search(cell):
        return "2σ"

    # # 15. εNd(0)
    # keyword = re.compile(r"(ε|ɛ|e|E)Nd ?\(0\)")
    # if keyword.search(cell):
    #     return "εNd(0)"

    # 16. εNd(t)
    # keyword = re.compile(r"(ε|ɛ|e|E)Nd ?\([tT0-9GMa]*\)")
    # if keyword.search(cell):
    #     return "εNd(t)"

    keyword = re.compile(r"(ε|ɛ|e|E)Nd")
    if keyword.search(cell):
        return "εNd(t)"

    # 17. fSm/Nd
    keyword = re.compile(r"fsm/?nd")
    if keyword.search(cell.lower()):
        return "fSm/Nd"

    # 18. TDM1
    if cell.find("TDM1") > -1:
        return "TDM1"

    # 19. TDM2
    if cell.find("TDM2") > -1:
        return "TDM2"

    # # 20. TDM
    # if cell.find("TDM") > -1:
    #     return "TDM"
    
    return ""

# with open("selected_paper.json", "r") as f:
#     all_marks = json.load(f)

# all_marks = ["1_" + item + ".pdf.txt" for item in all_marks]

all_marks = os.listdir(MARKS_DIR)
all_pdf = os.listdir(PDF_DIR)
all_excel = os.listdir(EXCEL_DIR)

excel_dict = dict()

mapping_dict = {
    "Sample": 0, "Nation/Region/GeoTectonic unit/Groups": 1, "Tectonic unit": 2, "Subtectonic unit/Sub groups": 3,
    "Longitude": 4, "Latitude": 5, "Lithology": 6, "Pluton": 7, "Age(Ma)": 8,
    "Sm": 9, "Nd": 10, "147Sm/144Nd": 11,
    "143Nd/144Nd": 12, "2σ": 13, "fSm/Nd": 14, "εNd(t)": 15,
    "TDM1": 16, "TDM2": 17, "Ref. Author": 18, "Ref. Year": 19, "Ref. Journal": 20, "Title": 21, 
    "Volume": 22, "Page": 23, "DOI": 24,
    "": -1
}

for excel in all_excel:
    if excel.count("_") != 4:
        continue
    pdf_name = "_".join(excel.split("_")[0: 2])
    if pdf_name not in excel_dict:
        excel_dict[pdf_name] = [excel]
    else:
        excel_dict[pdf_name].append(excel)

for pdf_name in excel_dict.keys():
    excel_dict[pdf_name] = sorted(excel_dict[pdf_name], key=lambda x:(int(x.split("_")[2]), int(x.split("_")[3])))



with open("pluton.json", "r") as f:
    plutons = set(json.load(f))

with open("lithology.json", "r") as f:
    lithologies = set(json.load(f))

with open("tectonic_unit.json", "r") as f:
    tectonic_units = set(json.load(f))

with open("geotectonic_unit.json", "r") as f:
    geotectonic_units = set(json.load(f))

with open("subtectonic_unit.json", "r") as f:
    subtectonic_units = set(json.load(f))

with open(META_DATA_DIR, "r") as f:
    meta_data = json.load(f)


def merge_single_file(params):
    mark_file = params[0]
    if mark_file[0] == "0":
        return []
    
    pdf_name = mark_file.split(".")[0][2:]
    
    with open(os.path.join(MARKS_DIR, mark_file), "r") as f:
        marks = json.load(f)
    
    # copyfile(os.path.join("../raw_pdf_exp3", pdf_name + ".pdf"), os.path.join("../select_pdf_split1_sample", pdf_name + ".pdf"))
    if pdf_name not in excel_dict or pdf_name not in meta_data:
        return []

    paper_geo_tect, paper_tect, paper_sub_tect = get_tect_from_title(meta_data[pdf_name]["title"])

    paper_age = get_age(meta_data[pdf_name]["abstract"])

    temp_paper_geo_tect = matching_keyword(meta_data[pdf_name]["title"], geotectonic_units)
    if temp_paper_geo_tect == "":
        temp_paper_geo_tect = matching_keyword(meta_data[pdf_name]["abstract"], geotectonic_units)
    paper_geo_tect = temp_paper_geo_tect if temp_paper_geo_tect else paper_geo_tect

    temp_paper_tect = matching_keyword(meta_data[pdf_name]["title"], tectonic_units)
    if temp_paper_tect == "":
        temp_paper_tect = matching_keyword(meta_data[pdf_name]["abstract"], tectonic_units)
    if temp_paper_tect:
        if not paper_sub_tect:
            paper_sub_tect = paper_tect
        paper_tect = temp_paper_tect


    temp_paper_sub_tect = matching_keyword(meta_data[pdf_name]["title"], subtectonic_units)
    if temp_paper_sub_tect == "":
        temp_paper_sub_tect = matching_keyword(meta_data[pdf_name]["abstract"], subtectonic_units)
    paper_sub_tect = temp_paper_sub_tect if temp_paper_sub_tect else paper_sub_tect


    # if paper_geo_tect == "" and paper_tect == "" and paper_sub_tect == "":
    temp_paper_geo_tect, temp_paper_tect, temp_paper_sub_tect = get_raw_tect_from_title(meta_data[pdf_name]["title"])
    paper_geo_tect = temp_paper_geo_tect if not paper_geo_tect else paper_geo_tect
    paper_tect = temp_paper_tect if not paper_tect else paper_tect
    paper_sub_tect = temp_paper_sub_tect if not paper_sub_tect else paper_sub_tect

    # if paper_geo_tect == "" and paper_tect == "" and paper_sub_tect == "":
    temp_paper_geo_tect, temp_paper_tect, temp_paper_sub_tect = get_raw_tect_from_title(meta_data[pdf_name]["abstract"].split(".")[0])
    paper_geo_tect = temp_paper_geo_tect if not paper_geo_tect else paper_geo_tect
    paper_tect = temp_paper_tect if not paper_tect else paper_tect
    paper_sub_tect = temp_paper_sub_tect if not paper_sub_tect else paper_sub_tect

    paper_pluton = matching_keyword(meta_data[pdf_name]["title"], plutons)
    if paper_pluton == "":
        paper_pluton = matching_keyword(meta_data[pdf_name]["abstract"], plutons)

    paper_lithology = matching_keyword(meta_data[pdf_name]["title"], lithologies)
    if paper_lithology == "":
        paper_lithology = matching_keyword(meta_data[pdf_name]["abstract"], lithologies)

    if paper_pluton == "":
        paper_pluton = get_pluton_from_text(meta_data[pdf_name]["title"])
    if paper_pluton == "":
        paper_pluton = get_pluton_from_text(meta_data[pdf_name]["abstract"])

    paper_title = meta_data[pdf_name]["title"]
    paper_journal = meta_data[pdf_name]["journal"]
    paper_year = str(meta_data[pdf_name]["year"])
    paper_page = str(meta_data[pdf_name]["first_page"]) + "-" + str(meta_data[pdf_name]["last_page"])
    paper_doi = meta_data[pdf_name]["doi"]
    paper_volume = str(meta_data[pdf_name]["volume"])
    if len(meta_data[pdf_name]["author"]["name"]) > 0:
        paper_author = meta_data[pdf_name]["author"]["name"][0] + " et al."
    else:
        paper_author = ""

    return_lines = []
    sample_data_dict = dict()
    sample_info_eids = []

    for eid, excel in enumerate(excel_dict[pdf_name]):
        excel_path = os.path.join(EXCEL_DIR, excel)
        sm_nd_table_indices = (-1, -1)
        if check_excel(excel_path) != 0:
            continue

        table = xlrd.open_workbook(excel_path).sheets()[0]
        col_num = table.ncols
        row_num = table.nrows

        # Check whether has Sm-Nd
        break_flag = False
        for rid in range(0, min(row_num, 3)):
            for cid in range(col_num):
                cell = str(table.cell_value(rid, cid))
                if check_sm_nd(cell):
                    sm_nd_table_indices = (rid, cid)
                    break_flag = True
                    break
            if break_flag:
                break
        
        # Process later if no Sm-Nd
        if sm_nd_table_indices[0] == -1:
            sample_info_eids.append(eid)
            continue

        # Get column headers
        headers_seq = []
        pluton_may_in_row, sample_cid = True, -1
        for cid in range(0, col_num):
            cell = str(table.cell_value(sm_nd_table_indices[0], cid))
            header_type = get_header_type(cell)
            if header_type == "Pluton":
                pluton_may_in_row = False
            if header_type == "Sample":
                sample_cid = cid
            headers_seq.append(header_type)
        
        # Must have Sample column
        if sample_cid == -1:
            continue
        
        # traverse all lines
        row_pluton = None
        for rid in range(sm_nd_table_indices[0] + 1, row_num):
            sm_nd_cell = str(table.cell_value(rid, sm_nd_table_indices[1]))
            if check_value(sm_nd_cell):
                sample_cell = str(table.cell_value(rid, sample_cid)).strip()
                if sample_cell not in sample_data_dict:
                    sample_data_dict[sample_cell] = dict()
                for cid in range(0, col_num):
                    sample_data_dict[sample_cell][headers_seq[cid]] = str(table.cell_value(rid, cid))

                    if headers_seq[cid] == "" and "Lithology" not in sample_data_dict[sample_cell]:
                        if str(table.cell_value(rid, cid)).lower() in lithologies:
                            sample_data_dict[sample_cell]["Lithology"] = str(table.cell_value(rid, cid))

                if pluton_may_in_row and row_pluton is not None:
                    sample_data_dict[sample_cell]["Pluton"] = row_pluton
            else:
                if pluton_may_in_row:
                    possible_pluton = str(table.cell_value(rid, sample_cid))
                    if possible_pluton.lower() in plutons:
                        row_pluton = possible_pluton
    
    # print(sample_data_dict)
    
    sample_info_eids = set(sample_info_eids)
    for eid, excel in enumerate(excel_dict[pdf_name]):
        if eid not in sample_info_eids:
            continue
        excel_path = os.path.join("../output/excel", excel)

        table = xlrd.open_workbook(excel_path).sheets()[0]
        col_num = table.ncols
        row_num = table.nrows

        sample_name_list = []
        for rid in range(row_num):
            for cid in range(col_num):
                if rid > 3 and cid > 3:
                    continue
                cell = str(table.cell_value(rid, cid))
                if cell.strip() in sample_data_dict.keys():
                    sample_name_list.append([rid, cid, cell])

        if len(sample_name_list) < 2:
            continue
        
        if sample_name_list[0][0] == sample_name_list[1][0]:
            same_row = True
        elif sample_name_list[0][1] == sample_name_list[1][1]:
            same_row = False
        else:
            continue
        
        headers_seq = []
        if same_row:
            for rid in range(sample_name_list[0][0] + 1, row_num):
                cell = str(table.cell_value(rid, 0))
                header_type = get_header_type(cell)
                headers_seq.append(header_type)
            
            for sample_name in sample_name_list:
                current_cid = sample_name[1]
                for rid in range(sample_name_list[0][0] + 1, row_num):
                    header_name = headers_seq[rid - (sample_name_list[0][0] + 1)]
                    sample_data_dict[sample_name[2].strip()][header_name] = str(table.cell_value(rid, current_cid))
        
        else:
            for cid in range(sample_name_list[0][1] + 1, col_num):
                cell = str(table.cell_value(0, cid))
                header_type = get_header_type(cell)
                headers_seq.append(header_type)

            for sample_name in sample_name_list:
                current_rid = sample_name[0]
                for cid in range(sample_name_list[0][1] + 1, col_num):
                    header_name = headers_seq[cid - (sample_name_list[0][1] + 1)]
                    sample_data_dict[sample_name[2].strip()][header_name] = str(table.cell_value(current_rid, cid))
    
    for sample_data in sample_data_dict.keys():
        single_line = ["" for _ in range(25)]
        for item in sample_data_dict[sample_data].items():
            single_line[mapping_dict[item[0]]] = item[1]
            single_line[18] = paper_author
            single_line[19] = paper_year
            single_line[20] = paper_journal
            single_line[21] = paper_title
            single_line[22] = paper_volume
            single_line[23] = paper_page
            single_line[24] = paper_doi
            if single_line[1] == "":
                single_line[1] = paper_geo_tect
            if single_line[2] == "":
                single_line[2] = paper_tect
            if single_line[3] == "" and paper_sub_tect != paper_tect:
                single_line[3] = paper_sub_tect
            if single_line[6] == "":
                single_line[6] = paper_lithology
            if single_line[7] == "":
                single_line[7] = paper_pluton
            if single_line[8] == "":
                single_line[8] = paper_age
            single_line[12].replace("+", "±")
            if "±" in single_line[12]:
                single_line[12], single_line[13] = single_line[12].split("±")[0: 2]
            elif "(" in single_line[12] and ")" in single_line[12]:
                if single_line[12].find("(") < single_line[12].find(")"):
                    single_line[12], single_line[13] = single_line[12][: single_line[12].find("(")], single_line[12][single_line[12].find("(") + 1: single_line[12].find(")")]
            
            if "." in single_line[13]:
                zero_flag = True
                select_num = []
                for char in single_line[13]:
                    if char != ".":
                        if char != "0":
                            zero_flag = False
                            select_num.append(char)
                        elif char == "0":
                            if not zero_flag:
                                select_num.append(char)
                
                single_line[13] = "".join(select_num)
            single_line[12] = single_line[12].strip()
            single_line[13] = single_line[13].strip()
        # if single_line[9] == "":
        #     continue
        return_lines.append(single_line)
    return return_lines


def merge_multi_file():
    all_line_data = []

    task_list = [(mark_file, 0) for mark_file in all_marks]

    # random.shuffle(task_list)

    process_pool = Pool(PROCESS_NUM)

    with tqdm(total=len(task_list)) as progress_bar:
        for return_data in process_pool.imap(merge_single_file, task_list):
            all_line_data += return_data
            progress_bar.update(1)

    process_pool.close()
    process_pool.join()

    blank_counter = 0
    fill_counter = 0
    for line in all_line_data:
        for i in line:
            if i.strip():
                fill_counter += 1
            else:
                blank_counter += 1
    
    print(fill_counter, blank_counter, fill_counter / (fill_counter + blank_counter))

    p = pd.DataFrame(columns=[item[0] for item in sorted(mapping_dict.items(), key=lambda x: x[1]) if item[0] != ""], data=all_line_data)

    writer = pd.ExcelWriter(OUTPUT_FILE_NAME) 
    p.to_excel(writer, index=False) 
    writer.save()

merge_multi_file()
