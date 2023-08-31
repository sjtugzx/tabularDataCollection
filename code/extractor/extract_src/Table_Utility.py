from config import *
from Utils import *
import fitz
from PIL import Image
import  numpy as np
import re
import  PyPDF2
import cv2
import models
from Inner_line import detect_table_structure_frame,detect_table_structure_noframe
import subprocess
import Text_extractor


#生成xml文件，并返回地址
def get_xml_path(pdfpath,page,force=False):
    name=get_name_from_path(pdfpath)
    xmlpath = os.path.abspath(os.path.join(XML_DIR,name + "_{}_pdfminer.xml".format(page)))
    if os.path.exists(xmlpath) and force==False:
        if get_size(xmlpath,"KB")>3:
            return xmlpath
    if get_size(pdfpath,"MB")>30:
        return "太大"
    try:
        #要用到超时报错功能，必须用subprocess，而不是os.system
        os.system('python pdf2txt.py  -o "{}" -p {} -t xml "{}"'.format(xmlpath,str(page+1),pdfpath))
        # subprocess.run(['/home/agrandtree/anaconda3/envs/torch/bin/pdf2txt.py  -o "{}" -p {} -t xml "{}"'.format(xmlpath,str(page+1),pdfpath)],timeout=8,shell=True)
    except:
        print("精确提取超时，转换提取模式")
        try:
            os.system('python pdf2txt.py  -o "{}" -p {} -n -t xml "{}"'.format(xmlpath, str(page + 1), pdfpath))
            # subprocess.run(['/home/agrandtree/anaconda3/envs/torch/bin/pdf2txt.py  -o "{}" -p {} -n -t xml "{}"'.format(
            #     xmlpath, str(page + 1), pdfpath)], timeout=8, shell=True)
        except:
            return "超时"
    return xmlpath


def detect_table_structure(
        paper_id: str,
        meta,
) -> models.TableStructure:
    ocr_flag=meta["ocr_flag"]
    direction=meta["direction"]
    table_type=meta["table_type"]
    autothres=meta["autothres"]
    pdfpath= meta["pdfpath"]
    xmlpath=meta["xmlpath"]
    area=meta["area"]
    page=meta["page"]
    pix,_=get_table_outline_img(pdfpath, page, area)

    if table_type == "frame":
        # ret要么是None 要么是structure
        ret = detect_table_structure_frame(pix, area, autothres)
        if ret:
            pix0 = page.getPixmap()
            rownum = len(ret.rows)
            colnum = len(ret.columns)
            tablecontent = [[""] * (colnum - 1)] * (rownum - 1)
            tablemerge = []
            rowid = colid = 0
            for row in ret.cells:
                for col in row:
                    if col.row_begin - col.row_end != 1 or col.column_begin - col.column_end != 1:
                        tablemerge.append({"row": rowid, "col": colid, "rowspan": col.row_begin - col.row_end,
                                           "colspan": col.column_begin - col.column_end})
                    rowid += col.row_begin - col.row_end
                    colid += col.column_begin - col.column_end
            return ret

    print("进行无框线表格处理")
    ret = detect_table_structure_noframe(pix,autothres, area)
    if ret:
        return ret
    else:
        raise RuntimeError("UNKNOWN ERROR")


def get_table_outline_img(pdfpath, page, area, density=2000, propotion = -1):
    doc = fitz.open(pdfpath)
    page = doc[page]
    pix = page.getPixmap()
    pdfheight = pix.height
    pdfwidth = pix.width
    clip = fitz.Rect([area[0] * pdfwidth, area[1] * pdfheight, area[2] * pdfwidth,
                      area[3] * pdfheight])
    if propotion<0:
        pdfzoom = max(density / pdfheight, density / pdfwidth)
    else:
        pdfzoom=propotion
    mat = fitz.Matrix(pdfzoom, pdfzoom).preRotate(0)
    pix = page.getPixmap(matrix=mat, alpha=False, clip=clip)
    pix = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pix = np.array(pix)
    return pix[:, :, ::-1],pdfzoom


def get_table_type(pdfpath,page,area):
    table_img,_=get_table_outline_img(pdfpath, page, area)
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    check = np.ones((dilatedrow.shape)) * 255
    check = dilatedrow == check
    thres = len(np.where(check == True)[0])
    # print("线框检测:检测到线框像素:",thres)
    if thres > 2000:
        # print("线框检测:有线框表")
        print(get_name_from_path(pdfpath),page,"framed")
        return "framed"
    return "unframed"


def get_ocr_flag(xmlpath):
    if os.path.exists(xmlpath):
        xml = open(xmlpath, "r", encoding="utf-8").read()
        letter = re.findall("<text font=.*?bbox=\"(.*?)\".*?</text>", xml)
        if len(letter) == 0:
            return "notext"
        return "text"
    return "notext"

#返回表格的方向，目前只支持left和up这样的常见类型
def get_direction_by_calc(xmlpath,area):
    xml = open(xmlpath, "r", encoding="utf-8").read()
    def getpdfxmlshape(box):
        return [float(i) for i in box.split(",")]
    ori_size = getpdfxmlshape(re.findall("<page.*?bbox=\"(.*?)\".*?>", xml)[0])
    orishape = [ori_size[2], ori_size[3]]
    areabox=[orishape[i%2] * e for i,e in enumerate(area)]
    letter = re.findall("<text font=.*?bbox=\"(.*?)\".*?</text>", xml)
    bboxes=[]
    for box in letter:
        tmp = Text_extractor.transbbox(box, orishape[1])
        bboxes.append(tmp)
    x0, y0, x1, y1 = areabox
    size = len(bboxes)
    i = 0
    sid= rot = nrot = 0
    while i < size:
        bx0, by0, bx1, by1 = bboxes[i]
        if y0 > by1 or y1 < by0:
            i += 1
            continue
        elif y0 <= by0 and by1 <= y1:
            if x0 <= bx0 and x1 >= bx1:
                if bboxes[i][2]-bboxes[i][0] > bboxes[i][3]-bboxes[i][1]:
                    rot+=1
                else:
                    nrot+=1
                sid+=1
        i+=1

    if sid < 10:
        print("没有ocr过的表格，不能处理")
        return "up"
    elif rot>=nrot:
        print("check rotated !")
        return "left"
    else:
        return "up"


#判断盒子是否在区域内。
def judge_in_area(font_area,restrict_area):
    c1= font_area[0]>restrict_area[0]
    c2= font_area[1]>restrict_area[1]
    c3= font_area[2]<restrict_area[2]
    c4= font_area[3]<restrict_area[3]
    if c1&c2&c3&c4:
        return 1
    return 0


def get_auto_thres(pdfpath,page,area):
    _,pdfzoom=get_table_outline_img(pdfpath, page, area)
    xmlpath = get_xml_path(pdfpath, page, force=False)
    xml = open(xmlpath, "r", encoding="utf-8").read()
    ori_size = re.findall("<page.*?bbox=\"(.*?)\".*?>", xml)[0]
    ori_size=[float(i) for i in ori_size.split(",")][2:]
    print(ori_size)

    restrict_area=[area[i]*ori_size[i%2] for i in range(4) ]

    letter = re.findall("<text font=.*?bbox=\"(.*?)\".*?size=\"(.*?)\".*?</text>", xml)
    # letter = re.findall("<text font=.*?bbox=\"(.*?)\".*?</text>", xml)
    print("约束区域：",restrict_area)

    if len(letter) == 0:
        # 没有文字层凭经验判断
        AUTOTHRES = 50.55555
        thres = 1.4
    else:
        n = sum = 0
        for thing in letter:
            font_area=[float(i) for i in thing[0].split(",")]
            size=thing[1]
            if float(size)==0 or not judge_in_area(font_area,restrict_area):
                continue
            sum += float(size)
            n += 1
        thres = sum / n * 0.8
        AUTOTHRES = int(sum / n * 0.55* 1.65* pdfzoom)
    print(f"自动阈值:{AUTOTHRES}")
    return AUTOTHRES,thres


def get_area_direction(direction, area):
    x1=area[0]
    y1=area[1]
    x2=area[2]
    y2=area[3]
    if direction=="up":
        return area
    elif direction=="left":
        return [1-y2,x1,1-y1,x2]
    elif direction =="down":
        return [1-x2,1-y2,1-x1,1-y1]
    elif direction=="right":
        return [y1,1-x2,y2,1-x1]


#生成旋转的pdf，返回路径
def rotate_pdf(pdfpath,page,outpath,rot="left"):
    name=get_name_from_path(pdfpath)
    pdfout=os.path.join(outpath,name+"_{}_rot.pdf".format(page))
    if os.path.exists(pdfout):
        return pdfout
    pdf_in = open(pdfpath, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_in)
    pdf_writer = PyPDF2.PdfFileWriter()
    pagefile = pdf_reader.getPage(page)

    if rot=="left":
        pagefile.rotateClockwise(90)
    elif rot=="down":
        pagefile.rotateClockwise(180)
    elif rot=="up":
        return pdfpath
    elif rot=="right":
        pagefile.rotateClockwise(270)
    pdf_writer.addPage(pagefile)
    create_dir(outpath)
    pdf_out = open(pdfout, 'wb')
    pdf_writer.write(pdf_out)
    pdf_out.close()
    pdf_in.close()
    return pdfout


def get_and_set_table_meta(pdfpath,page,area):
    xmlpath = get_xml_path(pdfpath, page)
    # xmlpath =os.path.join(XML_DIR,get_name_from_path(pdfpath) + "_{}_pdfminer.xml".format(page))
    if not os.path.exists(xmlpath) or get_size(xmlpath,"KB")<=3:
        return { "error":True }
    ocr_flag = get_ocr_flag(xmlpath)
    print(ocr_flag)
    if  ocr_flag  =="notext":
        print("没有文字层")
        return{ "ocr_flag" : "notext","error":True }
    direction=get_direction_by_calc(xmlpath,area)

    if direction !="up":
        area = get_area_direction(direction,area)
        pdfpath = rotate_pdf(pdfpath,page,PDF_PRO_DIR,direction)
        print("rotated,",get_name_from_path(pdfpath),page)
        page=0

    xmlpath = get_xml_path(pdfpath, page)
    autothres,spacethres=get_auto_thres(pdfpath,page,area)
    table_type = get_table_type(pdfpath, page, area)
    return {
        "ocr_flag":ocr_flag,
        "area":area,
        "direction":direction,
        "table_type":table_type,
        "autothres":autothres,
        "spacethres":spacethres,
        "pdfpath":pdfpath,
        "xmlpath":xmlpath,
        "page":page,
        "error": False
    }
