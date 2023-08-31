#一些用于测试 的 基本功能
import os
import cv2


def img_check(pic):
    cv2.imshow("win",pic)
    cv2.waitKey(0)


def img_save(pic,path,name):
    cv2.imwrite(os.path.join(path,name),pic)



def get_name_from_path(pdfpath):
    return ".".join(os.path.basename(pdfpath).split(".")[:-1])



def create_dir(path):
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)


def move_file(file,place,newname=""):
    if newname:
        os.system("cp {} {}".format(file,os.path.join(os.path.split(place)[0],newname)))
    else:
        os.system("cp {} {}".format(file, place))


#获取文件大小
def get_size(pdfpath,flag="MB"):
    if flag=="MB":
        return os.path.getsize(pdfpath)/1024/1024
    if flag=="B":
        return os.path.getsize(pdfpath)
    if flag=="KB":
        return os.path.getsize(pdfpath) /1024
    if flag=="GB":
        return os.path.getsize(pdfpath)/1024/1024 /1024

def get_file_num_of_dir(dir):
    if not os.path.exists(dir):
        return -1
    return len(os.listdir(dir))

def move(file,place,newname=""):
    if newname:
        os.system("cp {} {}".format(file,os.path.join(os.path.split(place)[0],newname)))
    else:
        os.system("cp {} {}".format(file, place))

