import cv2
import Utils
import numpy as np
import models
from typing import List


def detect_table_structure_frame(
        oripic: np.ndarray,
        area: List[float],
        AUTOTHRES: int):
    opt_pic = oripic.copy()
    gray = cv2.cvtColor(opt_pic, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 进行腐蚀处理
    merge = cv2.dilate(merge, kernel, iterations=3)
    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    check = np.ones((bitwiseAnd.shape)) * 255
    pointset = np.where(bitwiseAnd == check)
    pointset = [np.array([pointset[0][i], pointset[1][i]]) for i in range(len(pointset[0]))]
    if len(pointset) == 0:
        return None
    keypoints = []
    thres = AUTOTHRES / 1.3

    def distance(a, b):
        return np.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

    # 删除交点集中的冗余点，使所有冗余点归于其中心
    # n^2复杂度 ， 这一步可以优化
    for pointid, point in enumerate(pointset):
        flag = 0
        for centid, (centor, num) in enumerate(keypoints):
            if distance(point, centor) < thres:
                keypoints[centid] = [(centor * num + point) / (num + 1), num + 1]
                flag = 1
        if flag == 0:
            keypoints.append([point, 1])

    if len(keypoints) == 0:
        return None
    keypoints = [keypoints[i][0] for i in range(len(keypoints))]
    keypoints = np.array(keypoints, dtype=np.int)

    # 平行对齐
    last = keypoints[0][0]
    sum = last
    n = 1
    lastid = pointid = 0
    while pointid < len(keypoints):
        while pointid < len(keypoints) and keypoints[pointid][0] - last < 10:
            sum += keypoints[pointid][0]
            n += 1
            pointid += 1
        py = int(sum / n)
        for i in range(lastid, pointid):
            keypoints[i][0] = py
        if pointid >= len(keypoints):
            break
        lastid = pointid
        last = keypoints[pointid][0]
        sum = keypoints[pointid][0]
        n = 1
        pointid += 1
    keypoints = sorted(keypoints, key=lambda x: x[1])
    # print("按行排列:",keypoints)
    last = keypoints[0][1]
    sum = last
    n = 1
    lastid = pointid = 0
    while pointid < len(keypoints):
        while pointid < len(keypoints) and keypoints[pointid][1] - last < 10:
            sum += keypoints[pointid][1]
            n += 1
            pointid += 1
        px = int(sum / n)
        for i in range(lastid, pointid):
            keypoints[i][1] = px
        if pointid >= len(keypoints):
            break
        lastid = pointid
        last = keypoints[pointid][1]
        sum = keypoints[pointid][1]
        n = 1
        pointid += 1
    keyx = {}
    keyy = {}
    for points in keypoints:
        if points[1] not in keyx.keys():
            keyx[points[1]] = 1
        else:
            keyx[points[1]] += 1
        if points[0] not in keyy.keys():
            keyy[points[0]] = 1
        else:
            keyy[points[0]] += 1
    delx = []
    dely = []
    for x in keyx:
        if keyx[x] == 1:
            delx.append(x)
    for y in keyy:
        if keyy[y] == 1:
            dely.append(y)
    keypoints = [point for point in keypoints if (point[0] not in dely) and (point[1] not in delx)]
    keypoints = np.array(keypoints)
    if len(keypoints) == 0:
        return None
    keypoints = keypoints[:, ::-1]
    keypoints = [(p[0], p[1]) for p in keypoints]
    # 从这里开始keypoint的点坐标为 横,纵
    # print("对齐、删除孤立点、按行排列后的点个数为:", len(keypoints))
    # print("顶点个数:", len(keypoints))
    Ys = set()
    Xes = set()
    for point in keypoints:
        Ys.add(point[1])
        Xes.add(point[0])
    Xes = list(Xes)
    Xes.sort()
    Ys = list(Ys)
    Ys.sort()
    if len(Ys) <= 3 or len(Xes) <= 2:
        return None

    blocks = [[] for i in range(len(Ys) - 1)]

    # print("基本列数", len(Xes) - 1)
    # print("基本行数:", len(Ys) - 1)

    def judgeLine(pa, pb, thres=10):
        if pa[0] == pb[0]:
            for i in range(-1 * int(thres / 2), 1 * int(thres / 2)):
                if pa[0] + i >= merge.shape[1] or pa[0] + i < 0:
                    continue
                if np.sum(merge[pa[1]:pb[1], pa[0] + i]) / (pb[1] - pa[1] + 1) > 200:
                    return True
        if pa[1] == pb[1]:
            for i in range(-1 * int(thres / 2), 1 * int(thres / 2)):
                if pa[1] + i >= merge.shape[0] or pa[1] + i < 0:
                    continue
                if np.sum(merge[pa[1] + i, pa[0]:pb[0]]) / (pb[0] - pa[0] + 1) > 140:
                    return True
        return False

    for yid, y in enumerate(Ys):
        for xid, x in enumerate(Xes):
            if (x, y) not in keypoints:
                continue
            if Xes.index(x) == len(Xes) - 1 or Ys.index(y) == len(Ys) - 1:
                continue
            flag = 0
            for txid in range(xid + 1, len(Xes)):
                if (Xes[txid], y) not in keypoints:
                    continue
                if not judgeLine((x, y), (Xes[txid], y)):
                    break
                for tyid in range(yid + 1, len(Ys)):
                    if (Xes[txid], Ys[tyid]) in keypoints:
                        if not judgeLine((Xes[txid], Ys[tyid - 1]), (Xes[txid], Ys[tyid])):
                            break
                        if not judgeLine((x, Ys[tyid - 1]), (x, Ys[tyid])):
                            break
                        if judgeLine((Xes[txid - 1], Ys[tyid]), (Xes[txid], Ys[tyid])):
                            blocks[yid].append([Xes[xid], Ys[yid], Xes[txid], Ys[tyid], txid - xid, tyid - yid, 1])
                            flag = 1
                            break
                if flag == 1:
                    break
    ret = models.TableStructure()
    ret.confirmed = False
    cellist = [[] for i in range(len(Ys) - 1)]
    for rowid, row in enumerate(blocks):
        for blockid, block in enumerate(row):
            cell = models.TableCell()
            cell.column_begin = Xes.index(block[0])
            cell.row_begin = Ys.index(block[1])
            cell.column_end = Xes.index(block[2])
            cell.row_end = Ys.index(block[3])
            cellist[rowid].append(cell)
    ret.cells = cellist
    ret.area = area
    ret.rows = [(i / opt_pic.shape[0]) for i in Ys]
    ret.columns = [(i / opt_pic.shape[1]) for i in Xes]
    return ret


def detect_table_structure_noframe(
        oripic: np.ndarray,
        AUTOTHRES: int,
        area: List[float],
        fix_mode=0,
        fix_data=None
):
    if fix_mode == 1:
        row_pixel_thres = 255 * 12
    else:
        row_pixel_thres = 0
    # 根据thres确定的线宽,表格线删除需要 确定线宽,之后的划线也用到了线宽
    linewith = max(1, int(AUTOTHRES / 4))
    # print("自动线宽:", linewith)
    # self.pic是原图,所有后续操作不对原图进行, 而是对副本opt_pic进行优化,pic_show是为了展示效果的副本
    opt_pic = oripic.copy()
    W = opt_pic.shape[1]
    H = opt_pic.shape[0]
    # 将图像转为灰度图,并提取边缘.
    opt_pic = cv2.cvtColor(opt_pic, cv2.COLOR_BGR2GRAY)
    # # 在图片最下面画一条横线，以防模型没有截好
    # cv2.line(opt_pic, (0, H - 1), (W - 1, H - 1), 0, max(1, linewith))
    # # 在图片最上面画一条横线，以防图片没有竖线
    # cv2.line(opt_pic, (0,0), (W - 1, 0), 0, max(1, linewith))
    # edge = cv2.Canny(opt_pic, 20, 40, apertureSize=3)
    # # 通过边缘来确定表格中的直线以及端点
    # lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=220, minLineLength=300, maxLineGap=max(linewith, 3))
    # lines = [lines[i][0] for i in range(len(lines))]
    # # update 5.17 不用在此处去除线条
    # # 擦除表格中原有的划线.并确定表格位置, box中的元素依次是表格的 右左下上
    # box = [-99999, 99999, -99999, 99999]
    # for x1, y1, x2, y2 in lines:
    #     # 右x
    #     box[0] = max(x1, x2, box[0])
    #     # 左x
    #     box[1] = min(x1, x2, box[1])
    #     # 下y
    #     box[2] = max(y1, y2, box[2])
    #     # 上y
    #     box[3] = min(y1, y2, box[3])
    XMIN = 0
    XMAX = W - 1
    YMAX = H - 1
    YMIN = 0

    # update5.17
    # 完全去除线
    # 二值化
    mat, opt_pic = cv2.threshold(opt_pic, 200, 255, cv2.THRESH_BINARY)
    opt_pic = 255 - opt_pic
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    eroded = cv2.erode(opt_pic, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # 识别竖线
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    eroded = cv2.erode(opt_pic, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 进行腐蚀处理
    merge = cv2.dilate(merge, kernel, iterations=3)
    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(opt_pic, merge)
    opt_pic = 255 - merge2
    # 对优化图片进行腐蚀操作以及二值化以方便处理
    # ks是通过阈值算出来的核大小
    ks = max(2, int(AUTOTHRES / 6))
    # print("自动腐蚀处理阈值:", ks)
    kernel = np.ones((ks, ks), np.uint8)
    opt_pic = cv2.erode(opt_pic, kernel, iterations=1)
    mat, opt_pic = cv2.threshold(opt_pic, 200, 255, cv2.THRESH_BINARY)
    # 中值滤波,联结色块
    ks = int(ks / 2)
    ks = max(3, ks if ks % 2 == 1 else ks - 1)
    print("自动模糊处理阈值:", ks)
    opt_pic = cv2.medianBlur(opt_pic, ks)
    # 取反色,这样所有空白区域的亮度值加起来是0
    opt_pic = 255 - opt_pic
    # 划横线, 先对表格纵向做一定的扩张,以确定更多的横线
    xmin = XMIN
    ymin = YMIN
    xmax = XMAX
    ymax = YMAX
    # 记录下我们要划的线的两个端点
    linetodrawx = []
    # 我们接下来一行行扫描图像以确定划分表格的横线.
    # 若遇到某一行所有像素的亮度值之和为0,那么说明这一行没有元素,
    # 直到找到下一个像素值求和不为0的行,取这两行中间作为划分表格两行的分界线,
    # 并记录直线的两个端点(x1,y1,x2,y2)在linetodraw数组内
    Ys = []
    while ymin <= ymax:
        if (sum(opt_pic[ymin][xmin:xmax])) <= row_pixel_thres:
            liney = float(ymin)
            ymin += 1
            while ymin <= ymax and sum(opt_pic[ymin][xmin:xmax]) <= row_pixel_thres:
                ymin += 1
                liney += 0.5
            liney = int(liney + 0.5)
            Ys.append(liney)
            linetodrawx.append([XMIN, liney, XMAX, liney])
        ymin += 1
    print(Ys)
    if fix_mode == 2:
        linetodrawx = []
        Ys = fix_data
        for item in Ys:
            linetodrawx.append([XMIN, item, XMAX, item])
    print(Ys)
    ylines = [y1 for x1, y1, x2, y2 in linetodrawx]

    # print("自动分割阈值:",AUTOTHRES)

    # 根据阈值来切割表格横行, boxes是所有空白框,boxes[row][id]=[左,上,右,下]
    if fix_mode == 3:
        AUTOTHRES = AUTOTHRES / 2

    boxes = [[] for i in range(0, len(ylines) - 1)]
    for id, Y in enumerate(ylines):
        if id + 1 == len(ylines):
            break
        xmin = XMIN
        xmax = XMAX
        while xmin <= xmax:
            while xmin <= xmax and sum(opt_pic[ylines[id]:ylines[id + 1] + 1, xmin]) > 0:
                xmin += 1
            if xmin <= xmax and sum(opt_pic[ylines[id]:ylines[id + 1], xmin]) == 0:
                left = int(xmin)
                xmin += 1
                THRES = AUTOTHRES
                while xmin <= xmax and sum(opt_pic[ylines[id]:ylines[id + 1] + 1, xmin]) == 0:
                    xmin += 1
                    THRES -= 1
                if THRES <= 0:
                    right = int(xmin - 1)
                    boxes[id].append([left, ylines[id], right, ylines[id + 1]])

    # 展示过程 opt_pic1描绘了所有盒子和横线
    opt_pic1 = opt_pic.copy()
    opt_pic1 = cv2.cvtColor(opt_pic1, cv2.COLOR_GRAY2BGR)
    for row in boxes:
        for x1, y1, x2, y2 in row:
            cv2.rectangle(opt_pic1, (x1, y1), (x2, y2), (255, 255, 255), max(1, int(linewith / 3)))
    for x1, y1, x2, y2 in linetodrawx:
        cv2.line(opt_pic1, (x1, y1), (x2, y2), (0, 255, 255), max(1, int(linewith / 3)))
    # self.imgSave(opt_pic1, "step_2")

    # 区分文字和空白区域块,空白区域 按行blocks[row][id]=[左,上,右,下,类型],type=0:blank ,1: content
    blocks = [[] for i in range(0, len(boxes))]
    # print("空白盒子:",boxes)
    for rowid, row in enumerate(boxes):
        lastx = XMIN
        if len(row) == 0:
            continue
        if row[0][0] == XMIN:
            blocks[rowid].append([XMIN, boxes[rowid][0][1], row[0][2], boxes[rowid][0][3], 0])
            lastx = row[0][2]
            for x1, y1, x2, y2 in row[1:]:
                blocks[rowid].append([lastx, boxes[rowid][0][1], x1, boxes[rowid][0][3], 1])
                blocks[rowid].append([x1, boxes[rowid][0][1], x2, boxes[rowid][0][3], 0])
                lastx = x2
        else:
            lastx = XMIN
            for x1, y1, x2, y2 in row:
                blocks[rowid].append([lastx, boxes[rowid][0][1], x1, boxes[rowid][0][3], 1])
                blocks[rowid].append([x1, boxes[rowid][0][1], x2, boxes[rowid][0][3], 0])
                lastx = x2
        if lastx == XMAX:
            pass
        else:
            if blocks[rowid][-1][2] == 1:
                blocks[rowid].append([lastx, boxes[rowid][0][1], XMAX, boxes[rowid][0][3], 0])
            else:
                blocks[rowid].append([lastx, boxes[rowid][0][1], XMAX, boxes[rowid][0][3], 1])

    # 合并连通区域,
    # 先设定检查表,查看空白的存在情况
    # connects[row][id]=[rowbegin, rowend, xmin, xmax,rows ]
    connects = []
    check = [[] for i in range(0, len(boxes))]
    for rowid, row in enumerate(boxes):
        for box in row:
            check[rowid].append(1)
    for rowid, row in enumerate(boxes):
        for boxid, box in enumerate(row):
            if check[rowid][boxid] == 0:
                continue
            check[rowid][boxid] = 0
            rowbegin = rowid
            rowend = rowid + 1
            rowtmp = rowid + 1
            xmin, ymin, xmax, ymax = boxes[rowid][boxid]
            while rowtmp < len(boxes):
                flag = 0
                for tmpid, nextbox in enumerate(boxes[rowtmp]):
                    x1, y1, x2, y2 = nextbox
                    # if x1>=xmax or x2<=xmin or check[rowtmp][tmpid]==0:
                    if x1 >= xmax or x2 <= xmin or check[rowtmp][tmpid] == 0:
                        continue
                    if x1 >= xmin and x2 <= xmax:
                        xmin = x1
                        xmax = x2
                    elif x2 > xmax:
                        if x1 > xmin:
                            xmin = x1
                    elif x1 < xmin:
                        if x2 < xmax:
                            xmax = x2
                    rowend += 1
                    check[rowtmp][tmpid] = 0
                    flag = 1
                    break
                if rowend == len(boxes):
                    connects.append([rowbegin, rowend, xmin, xmax, rowend - rowbegin])
                    break
                elif flag == 1:
                    rowtmp += 1
                else:
                    connects.append([rowbegin, rowend, xmin, xmax, rowend - rowbegin])
                    break
    # print("联通个数:", len(connects))
    if len(connects) == 0:
        return None
        # raise ValueError("表格识别是错误的!")
    # 开始找通过最多连通分量的竖线横坐标
    linetodrawy = []
    # 判断某个联通区域是否还需要划分,1是需要,0是不需要
    check = [1 for thing in connects]
    # 只有一行的联通区域不需要划分了
    for id, thing in enumerate(connects):
        if thing[4] == 1:
            check[id] = 0

    def _getMaxPassInRegion(xmin, xmax, connects):
        maxpass = 0
        res = (xmin + xmax) / 2
        for x in range(xmin, xmax + 1):
            tmp = 0
            for con in connects:
                if x > con[2] and x < con[3]:
                    tmp += con[4]
            if tmp > maxpass:
                maxpass = tmp
                res = x
            if tmp == maxpass:
                res += 0.5
        return int(res), maxpass

    # 记录不必要查找的联通分量
    def _writeCheck(x, connects, check):
        for id, con in enumerate(connects):
            if con[2] <= x <= con[3]:
                check[id] = 0

    # Xes是划分的贯穿纵线的横坐标集合
    Xes = []
    for conid, con in enumerate(connects):
        if check[conid] == 0:
            continue
        check[conid] = 0
        xmin, xmax = [con[2], con[3]]
        x, maxpass = _getMaxPassInRegion(xmin, xmax, connects)
        # print("可连通{}行".format(maxpass))
        Xes.append(x)
        linetodrawy.append([x, YMIN, x, YMAX])
        _writeCheck(x, connects, check)

    Xes.sort()
    print(Xes)
    if fix_mode == 4:
        check = [1 for thing in connects]
        # 只有一行的联通区域不需要划分了
        for id, thing in enumerate(connects):
            if thing[4] == 1:
                check[id] = 0

        Xes = fix_data[1:-1]
        linetodrawy = []
        for item in Xes:
            linetodrawy.append([item, YMIN, item, YMAX])
            _writeCheck(item, connects, check)
    print(Xes)


    # 如果画的线通过文字区域这个线就要断开
    linetodrawY = []
    Xref = [[] for i in range(len(blocks))]
    for rowid, row in enumerate(blocks):
        for block in row:
            for x in Xes:
                if block[0] <= x <= block[2]:
                    if block[4] == 0:
                        linetodrawY.append([x, boxes[rowid][0][1], x, boxes[rowid][0][3]])
                        Xref[rowid].append(x)
    if Xes[0] > XMIN:
        Xes.insert(0, XMIN)
    if Xes[-1] < XMAX:
        Xes.insert(len(Xes), XMAX)

    # 通过行和x坐标 来获取单元格类型(是否有文字)
    def _getUnitType(rowid, X1, X2):
        for x1, y1, x2, y2, t in blocks[rowid]:
            if X1 <= x1 and X2 >= x2:
                return t
        return 0

    def _valToIndex(val, Li):
        id = 0
        for thing in Li:
            if val == thing:
                return id
            else:
                id += 1
        return 9999

    Units = [[] for id in range(len(Ys) - 1)]

    

    for rowid, row in enumerate(Xref):
        for id, X in enumerate(row):
            if id == 0:
                if X > XMIN:
                    t = _getUnitType(rowid, XMIN, X)
                    merge = _valToIndex(X, Xes)
                    unit = [XMIN, Ys[rowid], X, Ys[rowid + 1], merge, 1, t]
                    Units[rowid].append(unit)
            if id == len(row) - 1:
                t = _getUnitType(rowid, X, XMAX)
                merge = _valToIndex(XMAX, Xes) - _valToIndex(X, Xes)
                unit = [X, Ys[rowid], XMAX, Ys[rowid + 1], merge, 1, t]
                Units[rowid].append(unit)
                break
            Xb = row[id + 1]
            t = _getUnitType(rowid, X, Xb)
            merge = _valToIndex(Xb, Xes) - _valToIndex(X, Xes)
            unit = [X, Ys[rowid], Xb, Ys[rowid + 1], merge, 1, t]
            Units[rowid].append(unit)

    # print(Units)

    # 美化table,删除前排的空白框
    flag = 0
    for rowid in range(len(Units)):
        # if len(Units[rowid])<7:
        #     continue
        if Units[rowid][0][6] != 0:
            flag = 1
    if flag == 0:
        print("删除第一列空白列")
        for rowid in range(len(Units)):
            Units[rowid] = Units[rowid][1:]
        Xes = Xes[1:]

    flag = 0
    for rowid in range(len(Units)):
        # if len(Units[rowid])<7:
        #     continue
        if Units[rowid][-1][6] != 0:
            flag = 1
    if flag == 0:
        print("删除最后一列空白列")
        for rowid in range(len(Ys) - 1):
            Units[rowid] = Units[rowid][0:-1]
        Xes = Xes[:-1]
    # 删除多余末尾的空白
    flag = 0
    for rowid, row in enumerate(Units):
        # if len(Units[rowid])<7:
        #     continue
        if row[-1][4] == 1 and row[-1][6] != 0:
            flag = 1
    if flag == 0:
        print("检测到尾列有冗余列,自动调整")
        for rowid, row in enumerate(Units):
            # if len(Units[rowid]) < 7:
            #     continue
            if row[-1][4] > 1:
                row[-1][4] = 1
            elif row[-1][4] == 1:
                row[-2][4] = 1
                row[-2][2] = row[-1][2]
                Units[rowid] = row[:-1]
    # 删除多余首位的空白
    flag = 0
    for rowid, row in enumerate(Units):
        # if len(Units[rowid])<7:
        #     continue
        if row[0][4] == 1 and row[0][6] != 0:
            flag = 1
    if flag == 0:
        print("检测到首列有冗余列,自动调整")
        for rowid, row in enumerate(Units):
            # if len(Units[rowid]) < 7:
            #     continue
            if row[0][4] > 1:
                row[0][4] = 1
            elif row[0][4] == 1:
                row[1][4] = 1
                row[1][0] = row[0][0]
                Units[rowid] = row[1:]
    
    # print(Units)

    ret = models.TableStructure()
    ret.confirmed = False
    cellist = [[] for i in range(len(Ys) - 1)]
    for rowid, row in enumerate(Units):
        for unitid, unit in enumerate(row):
            try:
                cell = models.TableCell()
                cell.column_begin = Xes.index(unit[0])
                cell.row_begin = Ys.index(unit[1])
                cell.column_end = Xes.index(unit[2])
                cell.row_end = Ys.index(unit[3])
                cellist[rowid].append(cell)
            except:
                pass
    ret.cells = cellist
    ret.area = area
    oriheight = opt_pic.shape[0] / (area[3] - area[1])
    oriwidth = opt_pic.shape[1] / (area[2] - area[0])
    # ret.rows = [area[1] + i / oriheight for i in Ys]
    ret.rows = [i / opt_pic.shape[0] for i in Ys]
    # ret.columns = [area[0] + i / oriwidth for i in Xes]
    ret.columns = [i / opt_pic.shape[1] for i in Xes]

    # print(ret.rows)
    # print(ret.cells)
    return ret
