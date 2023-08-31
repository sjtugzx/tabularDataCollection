#!/home/agrandtree/miniconda3/envs/python37new/bin/python3.7
import sys
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter
import xlwt
import re
import fitz
import numpy as np
from PIL import Image
import cv2
from config import *
from typing import List
import models
import os
import Table_Utility

# 原地缩小盒子,p从0-1 ,逐渐减小盒子到0
def shrinkBboxes(para, bboxes):
    def shrinkBbox(p, bbox):
        px = 1 / 2 * p * (bbox[2] - bbox[0])
        py = 1 / 2 * p * (bbox[3] - bbox[1])
        return [bbox[0] + px, bbox[1] + py, bbox[2] - px, bbox[3] - py, ]
    return [shrinkBbox(para, i) for i in bboxes]


def create_table_excel(
        paper_id: str,
        page:int,
        table_id: int,
        cells: List[List[models.TableCell]],
        texts: List[str],
        confirmed: bool = False
) -> str:
    workbook = xlwt.Workbook(encoding="utf-8")
    worksheet = workbook.add_sheet("table")
    style = xlwt.XFStyle()
    align = xlwt.Alignment()
    align.horz = xlwt.Alignment.HORZ_CENTER
    align.vert = xlwt.Alignment.VERT_CENTER
    style.alignment = align
    index = 0
    for rowid, row in enumerate(cells):
        tmpid = 0
        for colid, cell in enumerate(row):
            merge = cell.column_end - cell.column_begin
            mergey = cell.row_end - cell.row_begin
            # print(texts[index],merge,mergey)
            worksheet.write_merge(rowid, rowid + mergey - 1, tmpid, tmpid + merge - 1, texts[index], style)
            tmpid += merge
            index += 1
    savepath = os.path.join(TABLE_DIR,
                            "{}_{}_{}_raw.xls".format(paper_id, page,table_id)) if confirmed == False else os.path.join(
        TABLE_DIR, "{}_{}_{}_final.xls".format(paper_id, page,table_id))
    workbook.save(savepath)
    return savepath

# 将PDFminer xml中的边界框转为cv2类型的边界框变成float list
def transbbox(box, height):
    tmp = [float(i) for i in box.split(",")]
    tmp1 = [tmp[0], height - tmp[3], tmp[2], height - tmp[1]]
    return tmp1

def get_area_text(
        hash_value: str,
        pdfpath: str,
        page: int,
        xmlpath:str,
        areas: List[List[float]],
        spacethres,
) :
    xml = open(xmlpath, "r", encoding="utf-8").read()
    # 得到fitz出来的pdf尺寸
    # 每段加一个空格,这样提取出来的东西是有空格的
    torep = re.findall("(<text font=.*?\">(.*)</text>\n<text> </text>)", xml)
    spaceflag=False
    if len(torep)!=0:
        spaceflag=True
        for thing in torep:
            str = thing[0].replace("\n<text> </text>", "")
            strnew = str + "\n" + str.replace(">{}<".format(thing[1]), "> <")
            xml = xml.replace(str, strnew)

    def getpdfxmlshape(box):
        return [float(i) for i in box.split(",")]

    ori_size = getpdfxmlshape(re.findall("<page.*?bbox=\"(.*?)\".*?>", xml)[0])
    orishape = [ori_size[2], ori_size[3]]
    doc =fitz.open(pdfpath)
    pixo =doc[page].getPixmap()
    h = pixo.height
    w= pixo.width
    for id in range(len(areas)):
        areas[id][0] = orishape[0] * areas[id][0]
        areas[id][1] = orishape[1] * areas[id][1]
        areas[id][2] = orishape[0] * areas[id][2]
        areas[id][3] = orishape[1] * areas[id][3]
    # print(areas)
    letter = re.findall("<text.*?bbox=\"(.*?)\".*?>(.*?)</text>", xml)
    # print(letter)
    bboxes = []
    letters = []
    for box in letter:
        tmp = transbbox(box[0], orishape[1])
        tmp =[tmp[0]*w/orishape[0],tmp[1]*h/orishape[1],tmp[2]*w/orishape[0],  tmp[3]*h/orishape[1]]
        bboxes.append(tmp)
        letters.append(box[1])
    #update7.15
    if not spaceflag:
        print("处理bboxes，文字间隔：",spacethres,spacethres/2.3)
        boxid=0
        while boxid < len(bboxes) and boxid+1 < len(bboxes) :
            if abs(bboxes[boxid+1][0]-bboxes[boxid][2]) >spacethres/3 or abs(bboxes[boxid+1][3]-bboxes[boxid][3]) >spacethres/3:
                bboxes.insert(boxid+1,bboxes[boxid])
                letters.insert(boxid+1," ")
                boxid+=2
                continue
            boxid+=1

    bboxes = shrinkBboxes(0.5, bboxes)
    texts = []
    for region in areas:
        texts.append(ocr_single_unit_by_xml(region,bboxes,letters))
    bboxes = shrinkBboxes(-1, bboxes)
    return texts,bboxes,letters,orishape


def ocr_single_unit_by_xml(area,bboxes,letters)->str:
    x0, y0, x1, y1 = area
    size = len(bboxes)
    i = 0
    res = ""
    while i < size:
        bx0, by0, bx1, by1 = bboxes[i]
        if y0 > by1 or y1 < by0:
            i += 1
            continue
        elif y0 <= by0 and by1 <= y1:
            if x0 <= bx0 and x1 >= bx1:
                res += letters[i]
                i += 1
                continue
        i += 1
    return res


def ocr_by_img(img) -> str:
    pass

