import torch
import clip,os,random
from PIL import Image
import numpy as np
from tqdm import tqdm
import predict_feat
device = "cuda" if torch.cuda.is_available() else "cpu"
def getfeat(id,cid):
    basepath=r"data/tstdata/"+cid+r"/"+str(id)
    tmpimgs=os.listdir(basepath)
    imgs=[basepath+r"/"+img for img in tmpimgs]
    if len(imgs)<10:
        subimgs=imgs
        subimgs=subimgs+[subimgs[0] for i in range(10-len(subimgs))]
    else:
        subimgs=random.sample(imgs,10)
    batch_size = len(subimgs)
    gallery_list = subimgs
    xs = predict_feat.inference_samples(predict_feat.model, predict_feat.transform, batch_size, gallery_list)
    xs = xs[0].detach().cpu().numpy()
    x = np.mean(xs, axis=0)
    return x
def get_k_center(cid):
    ids=os.listdir(r"C:/work/aic2023/data/tstdata/"+str(cid))
    if not os.path.exists(r"C:/work/aic2023/data/kcenter_reid/"+str(cid)):
        os.mkdir(r"C:/work/aic2023/data/kcenter_reid/"+str(cid))
    for i in tqdm(range(len(ids))):
        id = ids[i]
        try:
            tmp=getfeat(id,cid)
            np.save(r"C:/work/aic2023/data/kcenter_reid/"+str(cid)+r"/"+str(id)+".npy",tmp)
        except:
            print("error=====",id)
            f=open("errorid.csv","a")
            f.write(str(cid)+","+str(id)+"\n")
            f.close()

if __name__=="__main__":
    get_k_center("c122")
