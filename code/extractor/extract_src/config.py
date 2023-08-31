PDF_DIR="../PDF"
PDF_PRO_DIR= "../output/pdf"
XML_DIR="../output/xml"
IMG_DIR="../output/img"
TABLE_DIR="../output/excel"
IMG_TEST="../output/imgtest"
CAPTION_DIR="../output/caption"
ENTITY_RESULT="../output/All_Entity.json"
LOG_DIR="../log"
SOURCE_INPUT="../input"
TMP_EXCEL_DIR ="../output/tmpexcel"
TMP_CAPTION_DIR ="../output/tmpcaption"
TMP_IMG_DIR ="../output/tmpimg"
OUTLINE_IMG_DIR="../output/outline"

#记录每次运行到的地方
LOG_START = 0

#[ [pdfpath,page], ....]
#各种原因导致XML无法提取的
ERROR_XML = []
#PDF本身没有OCR
ERROR_OCR = []
#表格框线识别错误的
ERROR_TABLE=[]
#标题没识别成功的
Error_Caption=[]

#提取表格成功的
SUCCESS=[]
#caption成功的
