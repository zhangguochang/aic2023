import numpy as np
import os, copy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
import os, copy, shutil
from tqdm import tqdm
import random
import sys
import new_clust_step1
sys.setrecursionlimit(100000)  # 例如这里设置为十万
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(1234)
def get_k_center_bat(id, cid):
    ret = np.load(r"data/kcenter/" + str(cid) + r"/" + str(id) + ".npy")
    return ret
def get_k_center(id, cid):
    ret = np.load(r"data/kcenter_reid/" + str(cid) + r"/" + str(id) + ".npy")
    return ret
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
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
def get_id_ct(cid):
    rets={}
    ids = os.listdir(r"data/tstdata/"+str(cid))
    for id in ids:
        rets[id]=len(os.listdir(r"data/tstdata/"+str(cid)+r"/"+str(id)))
    return rets
def get_two_idsets_ssim(idsets1,idsets2,id_cts,cid):
    cts1=[]
    for idx in idsets1:
        cts1.append(id_cts[idx])
    rates1=[float(cts1_i)/sum(cts1) for cts1_i in cts1]

    tmp_0 = get_k_center(idsets1[0], cid)


    feat1=np.zeros(tmp_0.shape[0])
    for i in range(len(idsets1)):
        idx=idsets1[i]
        tmp1=get_k_center(idx, cid)
        feat1=feat1+tmp1*rates1[i]

    cts2=[]
    for idy in idsets2:
        cts2.append(id_cts[idy])
    rates2=[float(cts2_i)/sum(cts2) for cts2_i in cts2]

    feat2=np.zeros(tmp_0.shape[0])
    for j in range(len(idsets2)):
        idy=idsets2[j]
        tmp2=get_k_center(idy, cid)
        feat2=feat2+tmp2*rates2[j]

    dist=eucliDist(feat1,feat2)
    return dist

def calcum_maxssim_idset(ii,good_rets,id_maxmin_framid,id_cts,cid,alpha_dist):
    results={}
    good_rets_maxframids,good_rets_minframids=get_good_rets_maxmin_framids(good_rets, id_maxmin_framid)
    ii_rets_maxframids=good_rets_maxframids[ii]
    ii_rets_minframids=good_rets_minframids[ii]
    tmpx=[]
    for i in range(len(ii_rets_maxframids)):
        tmpx=tmpx+list(range(ii_rets_minframids[i],ii_rets_maxframids[i]+1))
    for j in range(len(good_rets)):
        if ii==j:
            continue
        j_rets_maxframids = good_rets_maxframids[j]
        j_rets_minframids = good_rets_minframids[j]
        tmp=[]
        for jj in range(len(j_rets_minframids)):
            tmp=tmp+list(range(j_rets_minframids[jj],j_rets_maxframids[jj]+1))
        if len(tmpx+tmp)==len(list(set(tmpx+tmp))):
            results[j]=get_two_idsets_ssim(good_rets[ii],good_rets[j],id_cts,cid)
    if len(results)>0:
        nwrets = sorted(results.items(), key=lambda x: x[1], reverse=False)
        if nwrets[0][1]<=alpha_dist:
            return nwrets[0][0]
        else:
            return -1
    else:
        return -1

def get_good_rets_maxmin_framids(good_rets,id_maxmin_framid):
    good_rets_maxframids = []
    good_rets_minframids = []
    for good_ret in good_rets:
        maxframids = []
        minframids = []
        for idy in good_ret:
            max_d = id_maxmin_framid[idy]["maxframid"]
            min_d = id_maxmin_framid[idy]["minframid"]
            maxframids.append(max_d)
            minframids.append(min_d)
        maxframids.sort()
        minframids.sort()
        good_rets_maxframids.append(maxframids)
        good_rets_minframids.append(minframids)
    return good_rets_maxframids,good_rets_minframids
def get_recall_ids(good_rets,id_maxmin_framid,id_cts,cid,sid):
    rets={}
    if sid=="S001":
        alpha_value=35.0
    else:
        alpha_value=32.0    
    for i in tqdm(range(len(good_rets))):
        maxsim_id=calcum_maxssim_idset(i,good_rets,id_maxmin_framid,id_cts,cid,alpha_dist=alpha_value)#35
        if maxsim_id>-1:
            rets[i]=maxsim_id
    return rets
def tran_good_rets_new_clust(rets,good_rets):
    ind_label={}
    for i in range(len(good_rets)):
        if i in rets:
            j=rets[i]
            if i not in ind_label:
                if j in ind_label:
                    ind_label[i]=ind_label[j]
                else:
                    if len(ind_label)>0:
                        curren_max_value=max(list(ind_label.values()))
                        ind_label[i] =curren_max_value+1
                    else:
                        ind_label[i] = 0
        else:
            if i not in ind_label:
                if len(ind_label) > 0:
                    curren_max_value = max(list(ind_label.values()))
                    ind_label[i] = curren_max_value + 1
                else:
                    ind_label[i] = 0
    final_good_rets_dict={}
    for idnx,label in ind_label.items():
        if label not in final_good_rets_dict:
            final_good_rets_dict[label]=good_rets[idnx]
        else:
            tmp=final_good_rets_dict[label]
            tmp=tmp+good_rets[idnx]
            final_good_rets_dict[label]=list(set(tmp))
    return final_good_rets_dict
def write_result_to_demo(final_clust_dict,cid):
    for label, ids in tqdm(final_clust_dict.items()):
        if not os.path.exists(r"try_demo_data/"+str(label)):
            os.mkdir(r"try_demo_data/"+str(label))
        for id in ids:
            imgs = os.listdir(r"data/tstdata/"+cid+r"/" + id)
            for img in imgs:
                shutil.copy(r"data/tstdata/"+cid+r"/"+id+r"/"+img,r"try_demo_data/"+str(label))
def get_final_clust_dict_label_cts(id_cts,final_clust_dict):
    results={}
    for label,ids in final_clust_dict.items():
        mm_n=0
        for id in ids:
            mm_n=mm_n+id_cts[id]
        results[label]=mm_n
    return results
def get_final_clust_dict_label_centervector(id_cts,final_clust_dict,final_label_cts,cid):
    results={}
    for label,ids in final_clust_dict.items():
        cts=final_label_cts[label]

        feat_0 = get_k_center(ids[0], cid)

        tmp_feat=np.zeros(feat_0.shape[0])
        for id in ids:
            feat=get_k_center(id, cid)
            tmp_feat=tmp_feat+feat*(id_cts[id]/float(cts))
        results[label]=tmp_feat
    return results

def good_rets_main(cid,sid):
    good_rets, bad_rets = new_clust_step1.main(cid,sid)
    id_cts=get_id_ct(cid)
    id_maxmin_framid=get_id_maxmin_framid(cid)
    rets=get_recall_ids(good_rets, id_maxmin_framid, id_cts, cid,sid)
    final_clust_dict=tran_good_rets_new_clust(rets,good_rets)

    rets_2 = get_recall_ids(list(final_clust_dict.values()), id_maxmin_framid, id_cts, cid,sid)
    final_clust_dict2 = tran_good_rets_new_clust(rets_2,list(final_clust_dict.values()))

    rets_3 = get_recall_ids(list(final_clust_dict2.values()), id_maxmin_framid, id_cts, cid,sid)
    final_clust_dict3 = tran_good_rets_new_clust(rets_3, list(final_clust_dict2.values()))

    rets_4 = get_recall_ids(list(final_clust_dict3.values()), id_maxmin_framid, id_cts, cid,sid)
    final_clust_dict4 = tran_good_rets_new_clust(rets_4, list(final_clust_dict3.values()))
    

    final_label_cts=get_final_clust_dict_label_cts(id_cts, final_clust_dict4)

    final_label_centervectors=get_final_clust_dict_label_centervector(id_cts, final_clust_dict4, final_label_cts, cid)

    #write_result_to_demo(final_clust_dict4, cid)
    return final_clust_dict4,bad_rets,final_label_cts,final_label_centervectors



if __name__=="__main__":
    cid="c001"
    good_rets_main(cid,sid="S001")



