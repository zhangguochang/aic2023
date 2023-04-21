import numpy as np
import os, copy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
import os, copy, shutil
from tqdm import tqdm
import random
import sys

sys.setrecursionlimit(100000)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(1234)
def get_framid_ctid_boundingbox_dict(cid):
    lines = open(r"data/sub/" + str(cid) + "/testdata.csv", encoding="utf-8").readlines()
    rets = {}
    for line in lines:
        line = line.replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
        tmps = line.split(",")
        # cid=tmps[0]
        framid = int(tmps[1])
        xmin = int(tmps[2])
        ymin = int(tmps[3])
        width = int(tmps[4])
        height = int(tmps[5])
        if framid not in rets:
            rets[framid] = {0: [xmin, ymin, width, height]}
        else:
            xpxx = rets[framid]
            xplen = copy.deepcopy(len(xpxx))
            xpxx[xplen] = [xmin, ymin, width, height]
            rets[framid] = xpxx
    return rets


def get_framid_ctid_img_dict(cid):
    imgs = os.listdir(r"data/sub/" + str(cid) + "/images")
    rets = {}
    for img in imgs:
        framid = img.split("_")[1]
        ctid = img.split("_")[2]
        rets[framid + "_" + ctid] = img
    return rets


def get_k_center_bat(id, cid):
    ret = np.load(r"data/kcenter/" + str(cid) + r"/" + str(id) + ".npy")
    return ret
def get_k_center(id, cid):
    ret = np.load(r"data/kcenter_reid/" + str(cid) + r"/" + str(id) + ".npy")
    return ret


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


def getdata(cid):
    ids = os.listdir(r"data/tstdata/" + str(cid))
    rets = []
    for id in ids:
        feat = get_k_center(id, cid)
        rets.append(feat.tolist())
    return rets

def get_id_maxmin_framid(cid):
    rets={}
    ids = os.listdir(r"data/tstdata/"+str(cid))
    for id in ids:
        imgs=os.listdir(r"data/tstdata/"+str(cid)+r"/"+id)
        lsts=[]
        for img in imgs:
            framid=int(img.split("_")[1])
            lsts.append(framid)
        maxframid=max(lsts)
        minframid=min(lsts)
        rets[id]={"maxframid":maxframid,"minframid":minframid}
    return rets
def getcid_clust(cid,sid):
    x = getdata(cid)
    if sid=="S001":
        threshold_value=28.0
    else:
        threshold_value=20.0
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold_value, compute_full_tree=True,
                                      affinity='euclidean', linkage='ward').fit_predict(np.array(x))  # 28 ward
    y_pred = cluster
    labels = y_pred.tolist()
    ids = os.listdir(r"data/tstdata/" + str(cid))
    rets = {}
    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if label not in rets:
            rets[label] = [id]
        else:
            tmp = rets[label]
            tmp.append(id)
            rets[label] = tmp
    return rets


def maodun_pd(id_0,id_others,id_maxminframid):
    id_maxframid=id_maxminframid[id_0]['maxframid']
    id_minframid=id_maxminframid[id_0]['minframid']
    for idx in id_others:
        id_0_maxframid=id_maxminframid[idx]['maxframid']
        id_0_minframid=id_maxminframid[idx]['minframid']
        if (id_minframid>=id_0_minframid and id_minframid<=id_0_maxframid) or (id_maxframid>=id_0_minframid and id_maxframid<=id_0_maxframid):
            return True
    return False
def maodun_js(idlsts,id_maxminframid):
    if len(idlsts)==1:
        return idlsts,[]
    delete_lsts=[]
    save_lsts=[idlsts[0]]
    for i in range(1,len(idlsts)):
        ismaodun=maodun_pd(idlsts[i],save_lsts,id_maxminframid)
        if ismaodun:
            delete_lsts.append(idlsts[i])
        else:
            save_lsts.append(idlsts[i])
    return save_lsts,delete_lsts


def alert_rets(good_rets,id_maxmin_framid):
    nw_good_rets=[]
    delete_totals=[]
    for good_ret in good_rets:
        save_lst, delete_lst = maodun_js(good_ret,id_maxmin_framid)
        nw_good_rets.append(save_lst)
        delete_totals=delete_totals+delete_lst
    for delete_idx in delete_totals:
        nw_good_rets.append([delete_idx])
    return nw_good_rets

def main(cid,sid):
    id_maxmin_framid=get_id_maxmin_framid(cid)
    good_rets=[]
    bad_rets=[]
    rets = getcid_clust(cid,sid)
    nw_rets = {}
    for label, ids in tqdm(rets.items()):
        nn = 0
        for id in ids:
            imgs = os.listdir(r"data/tstdata/"+cid+r"/" + id)
            nn = nn + len(imgs)
        nw_rets[label] = nn
    nwrets = sorted(nw_rets.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(nwrets)):
        label = nwrets[i][0]
        ids = rets[label]
        if i<100:
            good_rets.append(ids)
        else:
            bad_rets.append(ids)
    good_rets=alert_rets(good_rets, id_maxmin_framid)
    bad_rets = alert_rets(bad_rets, id_maxmin_framid)
    return good_rets,bad_rets


if __name__ == "__main__":
    cid="c001"
    good_rets,bad_rets=main(cid,sid="S001")
