import os,copy,shutil
from tqdm import tqdm
import random,cv2
from skimage.metrics import structural_similarity as ssim
def get_ssim(imgpth1,imgpth2):
    image1 = cv2.imread(imgpth1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    image2 = cv2.imread(imgpth2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    xh=image1.shape[0]
    xw=image1.shape[1]
    image2 = cv2.resize(image2, (xw, xh), interpolation=cv2.INTER_AREA)
    sim = ssim(image1, image2)
    return sim
def get_framid_ctid_boundingbox_dict(cid):
    lines=open(r"C:/work/aic2023/data/sub/"+str(cid)+"/testdata.csv",encoding="utf-8").readlines()
    rets={}
    for line in lines:
        line=line.replace(" ","").replace("\t","").replace("\r","").replace("\n","")
        tmps=line.split(",")
        #cid=tmps[0]
        framid=int(tmps[1])
        xmin=int(tmps[2])
        ymin=int(tmps[3])
        width=int(tmps[4])
        height=int(tmps[5])
        if framid not in rets:
            rets[framid]={0:[xmin,ymin,width,height]}
        else:
            xpxx=rets[framid]
            xplen=copy.deepcopy(len(xpxx))
            xpxx[xplen]=[xmin,ymin,width,height]
            rets[framid]=xpxx
    return rets
def get_framid_ctid_img_dict(cid):
    imgs=os.listdir(r"C:/work/aic2023/data/sub/"+str(cid)+"/images")
    rets={}
    for img in imgs:
        framid=img.split("_")[1]
        ctid=img.split("_")[2]
        rets[framid+"_"+ctid]=img
    return rets
def get_two_bbox_jj_area(bbox1,bbox2):
    xmin1, ymin1, width1, height1=bbox1
    xmin2, ymin2, width2, height2 = bbox2
    fxmin=max([xmin1,xmin2])
    fymin=max([ymin1,ymin2])
    fxmax=min([xmin1+width1,xmin2+width2])
    fymax=min([ymin1+height1,ymin2+height2])
    fwidth=fxmax-fxmin
    fheight=fymax-fymin

    if fxmin in range(xmin1,xmin1+width1+1) and fxmax in range(xmin1,xmin1+width1+1) and fymin in range(ymin1,ymin1+height1+1) and fymax in range(ymin1,ymin1+height1+1) and fxmin in range(xmin2,xmin2+width2+1) and fxmax in range(xmin2,xmin2+width2+1) and fymin in range(ymin2,ymin2+height2+1) and fymax in range(ymin2,ymin2+height2+1):
        if fxmin<fxmax and fymin<fymax:
            pass
        else:
            return 0.0
    else:
        return 0.0
    return fwidth*fheight/(1.0*width1*height1)

def get_different_person(cid):
    frets={}
    framid_ctid_bbox=get_framid_ctid_boundingbox_dict(cid)
    framid_ctid_img=get_framid_ctid_img_dict(cid)
    pid=1000000
    mxfrid=max(framid_ctid_bbox.keys())
    if not os.path.exists(r"C:/work/aic2023/data/tstdata/"+str(cid)):
        os.mkdir(r"C:/work/aic2023/data/tstdata/" + str(cid))
    for i in tqdm(range(mxfrid)):
        try:
            bbox1s = framid_ctid_bbox[i]
        except:
            print('miss:i=======',i)
            continue
        if i==0:
            for mi in range(len(bbox1s)):
                #bbox1 = bbox1s[m]
                if not os.path.exists(r"C:/work/aic2023/data/tstdata/"+str(cid)+r"/"+str(pid+mi)):
                    os.mkdir(r"C:/work/aic2023/data/tstdata/"+str(cid)+r"/"+str(pid+mi))
                img=framid_ctid_img["0_"+str(mi)]
                shutil.copy(r"C:/work/aic2023/data/sub/"+str(cid)+"/images/"+img,r"C:/work/aic2023/data/tstdata/"+str(cid)+r"/"+str(pid+mi))
                frets["0_"+str(mi)]=pid+mi
        for mii in range(len(bbox1s)):
            if str(i)+"_"+str(mii) not in frets:
                if len(frets.values())==0:
                    current_v=-1
                else:
                    current_v = copy.deepcopy(max(frets.values()))
                frets[str(i) + "_" + str(mii)] = current_v + 1
                if not os.path.exists(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1)):
                    os.mkdir(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))
                img = framid_ctid_img[str(i) + "_" + str(mii)]
                shutil.copy(r"C:/work/aic2023/data/sub/" + str(cid) + "/images/" + img,r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))
        try:
            bbox2s=framid_ctid_bbox[i+1]
        except:
            print('miss:i+1====',i+1)
            continue
        for n in range(len(bbox2s)):
            bbox2 = bbox2s[n]
            tmp_ret={}
            for m in range(len(bbox1s)):
                bbox1 = bbox1s[m]
                rate=get_two_bbox_jj_area(bbox1, bbox2)
                imgpth1 = r"C:/work/aic2023/data/sub/"+str(cid)+r"/images/" + framid_ctid_img[str(i + 1) + "_" + str(n)]
                imgpth2 = r"C:/work/aic2023/data/sub/"+str(cid)+r"/images/" + framid_ctid_img[str(i) + "_" + str(m)]
                sim = get_ssim(imgpth1, imgpth2)

                if rate > 0.75 and sim > 0.4:
                    tmp_ret[m] = rate + sim
            if len(tmp_ret)>0:
                nky=max(tmp_ret, key=tmp_ret.get)
                if str(i)+"_"+str(nky)  in frets:
                    frets[str(i+1)+"_" + str(n)] = frets[str(i)+"_"+str(nky)]
                    img = framid_ctid_img[str(i+1)+"_" + str(n)]
                    shutil.copy(r"C:/work/aic2023/data/sub/" + str(cid) + "/images/" + img,
                                r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(frets[str(i)+"_"+str(nky)]))
                else:
                    if len(frets.values()) == 0:
                        current_v = -1
                    else:
                        current_v = copy.deepcopy(max(frets.values()))
                    frets[str(i + 1) + "_" + str(n)] = current_v + 1
                    if not os.path.exists(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1)):
                        os.mkdir(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))
                    img = framid_ctid_img[str(i + 1) + "_" + str(n)]
                    shutil.copy(r"C:/work/aic2023/data/sub/" + str(cid) + "/images/" + img,r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))
            else:
                if len(frets.values())==0:
                    current_v=-1
                else:
                    current_v=copy.deepcopy(max(frets.values()))
                frets[str(i + 1) + "_" + str(n)]=current_v+1
                if not os.path.exists(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1)):
                    os.mkdir(r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))
                img = framid_ctid_img[str(i+1)+"_" + str(n)]
                shutil.copy(r"C:/work/aic2023/data/sub/" + str(cid) + "/images/" + img,
                            r"C:/work/aic2023/data/tstdata/" + str(cid) + r"/" + str(current_v + 1))



#if __name__=="__main__":
#    get_different_person('c002')