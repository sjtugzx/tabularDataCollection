from Table_Utility import get_table_outline_img
import Text_extractor
from Utils import *
from config import *
import Inner_line
import cv2
import Caption, Table_Utility
import numpy as np
import xlrd
import random

import pdfplumber
import json

from multiprocessing import Pool
from tqdm import tqdm
import fitz

import re
from subsclassify import check_format, check_encode, check_mixed_lines, check_false_text



def check_header(cell):
    keyword = re.compile(r"(ε|ɛ|e|E)Nd ?\((\d+|t|T)\)|1?43 ?Nd|1?44 ?Nd|1?47 ?Sm|\(cid:\d+\)Nd|fSm\/Nd|Sm\/Nd")
    if keyword.search(cell):
        return True
    else:
        return False

table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    "snap_tolerance": 3,
    "snap_x_tolerance": 3,
    "snap_y_tolerance": 3,
    "join_tolerance": 3,
    "join_x_tolerance": 3,
    "join_y_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "keep_blank_chars": False,
    "text_tolerance": 3,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
    "intersection_tolerance": 3,
    "intersection_x_tolerance": 3,
    "intersection_y_tolerance": 3,
}

def get_excel_by_pdf(pdfpath, page, tableid, outline, fix_mode=0):
    pdfname = get_name_from_path(pdfpath)
    excelpath = os.path.join(TABLE_DIR, "{}_{}_{}_raw.xls".format(pdfname, page, tableid))
    captionpath = os.path.join(CAPTION_DIR, "{}_{}_{}.txt".format(pdfname, page, tableid))

    try:
        meta = Table_Utility.get_and_set_table_meta(pdfpath, page, outline)
    except Exception as e:
        # print(e)
        ERROR_XML.append([pdfname, page, tableid])
        return 1
    if meta["error"] == True:
        if "ocr_flag" in meta.keys():
            ERROR_OCR.append([pdfpath, page, tableid])
            return 1
        else:
            ERROR_XML.append([pdfpath, page, tableid])
            return 1
    pdfpath = meta["pdfpath"]
    autothres = meta["autothres"]
    spacethres = meta["spacethres"]
    xmlpath = meta["xmlpath"]
    area = meta["area"]
    page_pro = meta["page"]
    table_type = meta["table_type"]
    ori_img, _ = get_table_outline_img(pdfpath, page_pro, [0, 0, 1.0, 1.0], 600)
    ori_img = np.ascontiguousarray(ori_img, dtype=np.uint8)
    orihei = ori_img.shape[0]
    oriwid = ori_img.shape[1]
    cv2.rectangle(ori_img, (int(area[0] * oriwid), int(area[1] * orihei)),
                  (int(area[2] * oriwid), int(area[3] * orihei)), (0, 255, 0), 2)
    img_save(ori_img, IMG_TEST, "{}_{}_{}.jpg".format(pdfname, page, tableid))
    op_img, _ = get_table_outline_img(pdfpath, page_pro, area, propotion=6)
    table_structure = None

    # autothres /= 2
    fix_data = None
    # print("fix flag 1")
    if fix_mode == 2:
        with pdfplumber.open(pdfpath) as pdf:
            table_page = pdf.pages[page]
            fix_data = [int(item["top"]) * 6 for item in
                        table_page.debug_tablefinder(table_settings).edges if item["orientation"] == "h"]
            if len(fix_data) < 3:
                return 1

    if fix_mode == 4:
        with pdfplumber.open(pdfpath) as pdf:
            table_page = pdf.pages[page]
            fix_data = [int(item["x0"]) * 6 for item in
                        table_page.debug_tablefinder(table_settings).edges if item["orientation"] == "v"]
            # print(fix_data)
            if len(fix_data) < 3:
                return 1

    if table_type == "framed":
        try:
            table_structure = Inner_line.detect_table_structure_noframe(op_img, autothres, area, fix_mode, fix_data)
        except:
            table_structure = Inner_line.detect_table_structure_noframe(op_img, autothres, area, fix_mode, fix_data)
    else:
        try:
            table_structure = Inner_line.detect_table_structure_noframe(op_img, autothres, area, fix_mode, fix_data)
        except:
            pass
    if table_structure == None:
        ERROR_TABLE.append([pdfname, page, tableid])
        return 1
    rows = table_structure.rows
    columns = table_structure.columns
    areas_local = []
    for row in table_structure.cells:
        for col in row:
            areas_local.append(
                [columns[col.column_begin], rows[col.row_begin], columns[col.column_end], rows[col.row_end]])
    wid = area[2] - area[0]
    hei = area[3] - area[1]
    areas_global = []
    for rec in areas_local:
        areas_global.append(
            [area[0] + rec[0] * wid, area[1] + rec[1] * hei, area[0] + rec[2] * wid, area[1] + rec[3] * hei])

    for rec in areas_global:
        p1 = (int(oriwid * rec[0]), int(orihei * rec[1]))
        p2 = (int(oriwid * rec[2]), int(orihei * rec[3]))
        cv2.rectangle(ori_img, p1, p2, (0, 0, 255), 1)
    texts, bboxes, letters, pdfshape = Text_extractor.get_area_text(pdfname, pdfpath, page_pro, xmlpath, areas_global,
                                                                    spacethres)
    img_save(ori_img, IMG_TEST, "{}_{}_{}.jpg".format(pdfname, page, tableid))
    img = cv2.resize(ori_img, (int(pdfshape[0]), int(pdfshape[1])))
    caption, img = Caption.getCaption_715(area, pdfshape, bboxes, letters, img=img)
    img_save(img, IMG_TEST, "{}_{}_{}.jpg".format(pdfname, page, tableid))
    if caption != "":
        # print(caption)
        with open(os.path.join(CAPTION_DIR, "{}_{}_{}.txt".format(pdfname, page, tableid)), "w", encoding="utf8") as f:
            f.write(caption)
    else:
        Error_Caption.append([[pdfpath, page, tableid]])
    try:
        excelpath = Text_extractor.create_table_excel(pdfname, page, tableid, table_structure.cells, texts)
    except:
        # print("存表失败：", pdfname, page, tableid)
        ERROR_TABLE.append([pdfname, page, tableid])
        return 1
    if os.path.exists(excelpath):
        return excelpath


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
    return format_ok or encode_ok or mixed_lines_ok or text_ok, [format_ok, encode_ok, mixed_lines_ok, text_ok]


def extract_single_file(params):
    pdf_path, detect_path, ocr_dir, mark_dir = params
    if os.path.exists(os.path.join(mark_dir, "1_" + pdf_path.split("/")[-1] + ".txt")):
        return
    if os.path.exists(os.path.join(mark_dir, "0_" + pdf_path.split("/")[-1] + ".txt")):
        return
    with open(detect_path, "r") as f:
        lines = [(int(item.strip().split(":")[0].split("_")[-1]), json.loads(item.strip().split(":")[-1])) for item in f.readlines()]
    
    pdf_header_flag = False
    pdf_mark = []

    if len(lines) > 0:
        for page, areas in lines:
            if len(areas) == 0:
                continue
            for area_id, area in enumerate(areas):
                try_time = 0
                extract_flag = False
                header_flag = False
                excel_path = 1
                while try_time < 5:
                    # excelpath = get_excel_by_pdf(pdfpath, page, table_id, outline, 0)
                    try:
                        excel_path = get_excel_by_pdf(pdf_path, page, area_id, area, try_time)
                    except KeyboardInterrupt:
                        return
                    except:
                        pass
                    if excel_path != 1 and excel_path != 0:
                        excel_ok, error_codes = check_excel(excel_path)
                        if excel_ok == 0:
                            extract_flag = True
                            break
                        else:
                            table = xlrd.open_workbook(excel_path).sheets()[0]
                            col_num = table.ncols
                            row_num = table.nrows
                            for rid in range(row_num):
                                for cid in range(col_num):
                                    cell = str(table.cell_value(rid, cid))
                                    if check_header(cell):
                                        extract_flag = True
                                        break
                                if extract_flag:
                                    break
                            if extract_flag:
                                break
                    try_time += 1
                
                if extract_flag:
                    table = xlrd.open_workbook(excel_path).sheets()[0]
                    col_num = table.ncols
                    row_num = table.nrows

                    for rid in range(row_num):
                        for cid in range(col_num):
                            cell = str(table.cell_value(rid, cid))
                            # if cell.lower().find("table") > -1 or rid < 4 or cid < 4:
                            header_flag = check_header(cell) or header_flag
                pdf_mark.append(1 if header_flag else 0)
                pdf_header_flag = pdf_header_flag or header_flag
    
    mark_path = os.path.join(mark_dir, "{}_".format(1 if pdf_header_flag else 0) + pdf_path.split("/")[-1] + ".txt")
    with open(mark_path, "w") as f:
        json.dump(pdf_mark, f)
    return 0


def extract_multi_file(pdf_dir, detect_dir, ocr_dir, mark_dir, process_num):
    pdf_names = os.listdir(pdf_dir)
    pdf_paths = []
    detect_paths = []
    for pdf_name in pdf_names:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if os.path.exists(os.path.join(mark_dir, "1_" + pdf_path.split("/")[-1] + ".txt")):
            continue
        if os.path.exists(os.path.join(mark_dir, "0_" + pdf_path.split("/")[-1] + ".txt")):
            continue
        if not os.path.exists(os.path.join(detect_dir, pdf_name.replace(".pdf", ".txt"))):
            continue
        pdf_paths.append(pdf_path)
        detect_paths.append(os.path.join(detect_dir, pdf_name.replace(".pdf", ".txt")))
    
    task_list = [(pdf_paths[i], detect_paths[i], ocr_dir, mark_dir) for i in range(len(pdf_paths))]

    random.shuffle(task_list)

    process_pool = Pool(process_num)

    with tqdm(total=len(task_list)) as progress_bar:
        for return_code in process_pool.imap_unordered(extract_single_file, task_list):
            # write_file.write(json.dumps(return_chapter) + "\n")
            progress_bar.update(1)

    process_pool.close()
    process_pool.join()

if __name__ == '__main__':
    PDF_DIR = "../pdfs/"
    DETECT_DIR = "../detect_output"
    MARKS_DIR = "../marks"
    PROCESS_NUM = 30
    extract_multi_file(PDF_DIR, DETECT_DIR, "", MARKS_DIR, PROCESS_NUM)
