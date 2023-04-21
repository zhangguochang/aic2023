import os
import cv2
import glob
import os
from tqdm import tqdm
import step3,step4
def deleteimgs():
    imgs=os.listdir(r'C:/work/aic2023/yolov7/inference/images')
    for img in imgs:
        imgpath=r'C:/work/aic2023/yolov7/inference/images/'+img
        if os.path.exists(imgpath):
            os.remove(imgpath)
def step1(sid,cid):
    videoPath=r"C:/work/aic2023/data/test/"+sid+r"/"+cid+r"/video.mp4"
    cap = cv2.VideoCapture(videoPath)
    suc = cap.isOpened()
    frame_count = 0
    while suc:
        suc, frame = cap.read()
        cv2.imwrite(r'C:/work/aic2023/yolov7/inference/images/' + cid + "_" + str(frame_count) + ".jpg", frame)
        frame_count += 1
    cap.release()
def step2(camid):
    if not os.path.exists(r"C:/work/aic2023/data/sub/" + camid):
        os.mkdir(r"C:/work/aic2023/data/sub/" + camid)
    if not os.path.exists(r"C:/work/aic2023/data/sub/" + camid + r"/images"):
        os.mkdir(r"C:/work/aic2023/data/sub/" + camid + r"/images")
    cmd1 = "cd C:/work/aic2023/yolov7 && python detect.py  --conf 0.25  --source inference/images --camid " + camid
    os.system(cmd1)

def main():
    for cid in ["c001","c002","c003","c004","c005","c006","c007","c014","c015","c016","c017","c018","c019","c047","c048","c049","c050","c051","c052","c076","c077","c078","c079","c080","c081","c100","c101","c102","c103","c104","c105","c118","c119","c120","c121","c122","c123","c124","c125","c126","c127","c128","c129"]:
        if cid in ["c001","c002","c003","c004","c005","c006","c007"]:
            sid="S001"
        elif cid in ["c014","c015","c016","c017","c018","c019"]:
            sid="S003"
        elif cid in ["c047","c048","c049","c050","c051","c052"]:
            sid="S009"
        elif cid in ["c076","c077","c078","c079","c080","c081"]:
            sid="S014"
        elif cid in ["c100","c101","c102","c103","c104","c105"]:
            sid="S018"
        elif cid in ["c118","c119","c120","c121","c122","c123"]:
            sid="S021"
        elif cid in ["c124","c125","c126","c127","c128","c129"]:
            sid="S022"
        else:
            sid="S001"
        step1(sid,cid)
        step2(cid)
        step3.get_different_person(cid)
        step4.get_k_center(cid)
if __name__=="__main__":
    main()
