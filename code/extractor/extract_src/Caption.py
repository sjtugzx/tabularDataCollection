import cv2
import Text_extractor


def get_index_table(letters):
    id=0
    startlist=[]
    while id+5<len(letters):
        if letters[id:id+5]==["T","a","b","l","e"] or letters[id:id+5]==["T","A","B","L","E"]:
            startlist.append(id)
            id+=1
        id+=1
    return startlist

def getCaption_715(outline, pdfshape, bboxes, letters,img=None):
    outline=[outline[0]*pdfshape[0],outline[1]*pdfshape[1],outline[2]*pdfshape[0],outline[3]*pdfshape[1]]
    #选出最合适的Table标题的开始位置,目前规则:位置小于表格外框且离框越近的就认为是Table关键字
    scan_area=[]
    best=999999
    start = -1
    for start_candi in get_index_table(letters):
        if bboxes[start_candi][1] < outline[1]:
            if (outline[1]-bboxes[start_candi][1])/pdfshape[1] <best:
                start=start_candi
                best=(outline[1]-bboxes[start_candi][1])/pdfshape[1]
    if start>0:
        #标题位于 表格外框上面， 如果标题处于 外框上 pdf页0.1位置以时， 如果标题 位于外框左边0.1pdf页内，或者位于在外框正上方，那么就判定为 标题
        if bboxes[start][1] < outline[1] and  (outline[1]-bboxes[start][1])/pdfshape[1]<0.2 \
                and (abs(outline[0]-bboxes[start][0])/pdfshape[0]<0.1 or (outline[0]<bboxes[start][0]<outline[2])) :
            if bboxes[start][0] < outline[0]:
                scan_area=[bboxes[start][0], bboxes[start][1],outline[2]+(outline[0]-bboxes[start][0]),outline[1]]
            else:
                scan_area=[ outline[0], bboxes[start][1],outline[2],outline[1]  ]

    title = ""
    bboxes=Text_extractor.shrinkBboxes(0.5, bboxes)
    if scan_area!=[]:
        title=Text_extractor.ocr_single_unit_by_xml(scan_area,bboxes,letters)
        cv2.rectangle(img, [int(scan_area[0]), int(scan_area[1])], [int(scan_area[2]), int(scan_area[3])], (255, 0, 0),2)
    return title,img


