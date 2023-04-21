import numpy as np
import os, copy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
import os, copy, shutil
from tqdm import tqdm
import random
import sys
import new_clust_step2


from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
model_id = 'damo/cv_passvitb_image-reid-person_market'
image_reid_person = pipeline(Tasks.image_reid_person, model=model_id)


sys.setrecursionlimit(100000)  # 例如这里设置为十万


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


def get_key_by_value(dict_x, vli):
    for ky, vls in dict_x.items():
        if vls == vli:
            return ky
    return ''


def get_id_ct(cid):
    rets = {}
    ids = os.listdir(r"data/tstdata/" + str(cid))
    for id in ids:
        rets[id] = len(os.listdir(r"data/tstdata/" + str(cid) + r"/" + str(id)))
    return rets


def get_id_maxmin_framid(cid):
    rets = {}
    ids = os.listdir(r"data/tstdata/" + str(cid))
    for id in ids:
        imgs = os.listdir(r"data/tstdata/" + str(cid) + r"/" + id)
        lsts = []
        for img in imgs:
            framid = int(img.split("_")[1])
            lsts.append(framid)
        maxframid = max(lsts)
        minframid = min(lsts)
        rets[id] = {"maxframid": maxframid, "minframid": minframid}
    return rets


def pd_two_idsets_is_jj(idsets, idsets1, id_maxmin_framid):
    id_max_framids = []
    id_min_framids = []
    for idx in idsets:
        id_min_framids.append(id_maxmin_framid[idx]["minframid"])
        id_max_framids.append(id_maxmin_framid[idx]["maxframid"])
    id_min_framids.sort()
    id_max_framids.sort()
    id1_max_framids = []
    id1_min_framids = []
    for idx1 in idsets1:
        id1_min_framids.append(id_maxmin_framid[idx1]["minframid"])
        id1_max_framids.append(id_maxmin_framid[idx1]["maxframid"])
    id1_min_framids.sort()
    id1_max_framids.sort()
    tmp = []
    for i in range(len(id_min_framids)):
        tmp = tmp + list(range(id_min_framids[i], id_max_framids[i] + 1))
    tmp1 = []
    for j in range(len(id1_min_framids)):
        tmp1 = tmp1 + list(range(id1_min_framids[j], id1_max_framids[j] + 1))
    if len(tmp + tmp1) == len(list(set(tmp + tmp1))):
        return False
    else:
        return True


def combine_dicts(tmpxxt, tmpxxt1):
    rets = {}
    for kys, vls in tmpxxt.items():
        if kys in tmpxxt1.items():
            rets[kys] = tmpxxt[kys] + tmpxxt1[kys]
        else:
            rets[kys] = tmpxxt[kys]
    for kys1, vls1 in tmpxxt1.items():
        if kys1 not in rets:
            rets[kys1] = tmpxxt1[kys1]
    return rets


def get_tran_dict_kv(txpx_rets):
    tran_txpx_rets = {}
    for tx_ky, tx_vl in txpx_rets.items():
        if tx_vl not in tran_txpx_rets:
            tran_txpx_rets[tx_vl] = [tx_ky]
        else:
            tmm_nn = tran_txpx_rets[tx_vl]
            tmm_nn.append(tx_ky)
            tran_txpx_rets[tx_vl] = tmm_nn
    return tran_txpx_rets


def pd_str_cts_lst(str_labels):
    n = 0
    for str_label in str_labels:
        if "left" in str_label:
            n = n + 1
    if n > 1:
        return True
    else:
        return False


def pd_two_idsets_jj(idset, idset1, cid):
    id_maxmin_framid = get_id_maxmin_framid(cid)
    id_0_min_framids = []
    id_0_max_framids = []
    for idx in idset:
        id_0_min_framids.append(id_maxmin_framid[idx]["minframid"])
        id_0_max_framids.append(id_maxmin_framid[idx]["maxframid"])

    id_0_min_framids.sort()
    id_0_max_framids.sort()

    tmp = []
    for i in range(len(id_0_min_framids)):
        tmp = tmp + list(range(id_0_min_framids[i], id_0_max_framids[i] + 1))
    id_1_min_framids = []
    id_1_max_framids = []
    for idy in idset1:
        id_1_min_framids.append(id_maxmin_framid[idy]["minframid"])
        id_1_max_framids.append(id_maxmin_framid[idy]["maxframid"])
    id_1_min_framids.sort()
    id_1_max_framids.sort()
    tmp1 = []
    for j in range(len(id_1_min_framids)):
        tmp1 = tmp1 + list(range(id_1_min_framids[j], id_1_max_framids[j] + 1))
    if len(tmp + tmp1) == len(list(set(tmp + tmp1))):
        return False
    else:
        return True


def have_sm_cids_1(save_lsts, left_label1, n_good_rets_dict):
    dtc1 = n_good_rets_dict[left_label1]
    cids = list(dtc1.keys())
    for label in save_lsts:
        cidtmps = n_good_rets_dict[label]
        for cid, idset in cidtmps.items():
            if cid in cids:
                idset1 = dtc1[cid]
                is_jj = pd_two_idsets_jj(idset, idset1, cid)
                if is_jj:
                    return save_lsts, [left_label1]
    save_lsts.append(left_label1)
    return save_lsts, []


def get_md_labels_rets(str_labels, n_good_rets_dict):
    left_labels = []
    final_save_lsts = []
    for str_label in str_labels:
        if "left" in str_label:
            left_labels.append(int(str_label.split("_")[-1]))
        else:
            final_save_lsts.append(str_label)
    left_label0 = left_labels[0]
    save_lsts = [left_label0]
    delte_lsts = []
    for i in range(1, len(left_labels)):
        save_lsts, delte_lst = have_sm_cids_1(save_lsts, left_labels[i], n_good_rets_dict)
        delte_lsts = delte_lsts + delte_lst
    for save_lsti in save_lsts:
        final_save_lsts.append("cleft_" + str(save_lsti))
    final_delte_lsts = ["cleft_" + str(lbbb) for lbbb in delte_lsts]
    return final_save_lsts, final_delte_lsts


def get_no_md_rets(tran_txpx_rets, n_good_rets_dict):
    nw_rets = {}
    total_deletes = []
    for i, str_labels in tran_txpx_rets.items():
        if len(str_labels) > 1:
            is_sms = pd_str_cts_lst(str_labels)
            if is_sms:
                final_save_lsts, final_delte_lsts = get_md_labels_rets(str_labels, n_good_rets_dict)
                nw_rets[i] = final_save_lsts
                total_deletes = total_deletes + final_delte_lsts
                # ---------------------
            else:
                nw_rets[i] = str_labels
        else:
            nw_rets[i] = str_labels
    maxlabel_xp = max(list(nw_rets.keys()))
    maxlabel = copy.deepcopy(maxlabel_xp)
    for j in range(len(total_deletes)):
        nw_rets[maxlabel + j + 1] = [total_deletes[j]]
    hh_results = {}
    for iib, xxlbs in nw_rets.items():
        for xxlb in xxlbs:
            hh_results[xxlb] = iib
    return hh_results


def get_two_different_cid_clust(id_maxmin_framid, n_good_rets_dict, final_label_cts, final_label_centervectors, cid,
                                sid, alpha):
    good_rets_dict1, bad_rets_lsts1, final_label_cts1, final_label_centervectors1 = new_clust_step2.good_rets_main(cid,
                                                                                                                   sid)
    rets = {}
    for label, feat in final_label_centervectors.items():
        tmps = {}
        for label1, feat1 in final_label_centervectors1.items():
            if cid in n_good_rets_dict[label]:
                idsets = n_good_rets_dict[label][cid]
                idsets1 = good_rets_dict1[label1]
                is_jj = pd_two_idsets_is_jj(idsets, idsets1, id_maxmin_framid)
                if not is_jj:
                    tmps[label1] = eucliDist(feat, feat1)
            else:
                tmps[label1] = eucliDist(feat, feat1)

        nwtmps = sorted(tmps.items(), key=lambda x: x[1], reverse=False)
        if len(nwtmps) > 0:
            if nwtmps[0][1] <= alpha:
                rets[label] = nwtmps[0][0]
    txpx_rets = {}
    for label, feat in final_label_centervectors.items():
        if label not in rets:
            if len(txpx_rets) > 0:
                current_v = max(list(txpx_rets.values()))
                txpx_rets["cleft_" + str(label)] = current_v + 1
            else:
                txpx_rets["cleft_" + str(label)] = 0
        else:
            if "cleft_" + str(label) not in txpx_rets:
                tmplabel1 = rets[label]
                if "cright_" + str(tmplabel1) in txpx_rets:
                    txpx_rets["cleft_" + str(label)] = txpx_rets["cright_" + str(tmplabel1)]
                else:
                    if len(txpx_rets) > 0:
                        current_v = max(list(txpx_rets.values()))
                        txpx_rets["cleft_" + str(label)] = current_v + 1
                    else:
                        txpx_rets["cleft_" + str(label)] = 0
    for label1, feat1 in final_label_centervectors1.items():
        if label1 not in list(rets.values()):
            if len(txpx_rets) > 0:
                current_v = max(list(txpx_rets.values()))
                txpx_rets["cright_" + str(label1)] = current_v + 1
            else:
                txpx_rets["cright_" + str(label1)] = 0
        else:
            if "cright_" + str(label1) not in txpx_rets:
                kylabel = get_key_by_value(rets, label1)
                if "cleft_" + str(kylabel) not in txpx_rets:
                    if len(txpx_rets) > 0:
                        current_v = max(list(txpx_rets.values()))
                        txpx_rets["cright_" + str(label1)] = current_v + 1
                    else:
                        txpx_rets["cright_" + str(label1)] = 0
                else:
                    txpx_rets["cright_" + str(label1)] = txpx_rets["cleft_" + str(kylabel)]

    # -----------------------------
    tran_txpx_rets = get_tran_dict_kv(txpx_rets)
    print("<<<<<<<>>>>>>>>>>>>>>>>>>>>>>><<<<<<<>>>>>>",txpx_rets)
    txpx_rets = get_no_md_rets(tran_txpx_rets, n_good_rets_dict)

    f_good_rets_dict = {}
    f_final_label_cts = {}
    f_final_label_centervectors = {}
    print("<<<<<<<>>>-------------------->>><<<<<<<>>>>>>", txpx_rets)
    for kys, vlss in txpx_rets.items():
        if vlss not in f_good_rets_dict:
            if "cleft_" in kys:
                f_good_rets_dict[vlss] = n_good_rets_dict[int(kys.replace("cleft_", ""))]
            else:
                f_good_rets_dict[vlss] = {cid: good_rets_dict1[int(kys.replace("cright_", ""))]}
        else:
            tmpxxt = f_good_rets_dict[vlss]
            if "cleft_" in kys:
                tmpxxt1 = n_good_rets_dict[int(kys.replace("cleft_", ""))]
                tmpxxt = combine_dicts(tmpxxt, tmpxxt1)
                f_good_rets_dict[vlss] = tmpxxt
            else:
                tmpxxt[cid] = good_rets_dict1[int(kys.replace("cright_", ""))]
                f_good_rets_dict[vlss] = tmpxxt

    for kys, vlss in txpx_rets.items():
        if vlss not in f_final_label_cts:
            if "cleft_" in kys:
                f_final_label_cts[vlss] = final_label_cts[int(kys.replace("cleft_", ""))]
            else:
                f_final_label_cts[vlss] = final_label_cts1[int(kys.replace("cright_", ""))]
        else:
            tmpct = f_final_label_cts[vlss]
            if "cleft_" in kys:
                tmpct1 = final_label_cts[int(kys.replace("cleft_", ""))]
                tmpct = tmpct + tmpct1
                f_final_label_cts[vlss] = tmpct
            else:
                tmpct1 = final_label_cts1[int(kys.replace("cright_", ""))]
                tmpct = tmpct + tmpct1
                f_final_label_cts[vlss] = tmpct

    for kys, vlss in txpx_rets.items():
        if vlss not in f_final_label_centervectors:
            if "cleft_" in kys:
                f_final_label_centervectors[vlss] = {
                    "vector": [final_label_centervectors[int(kys.replace("cleft_", ""))]],
                    "cts": [final_label_cts[int(kys.replace("cleft_", ""))]]}
            else:
                f_final_label_centervectors[vlss] = {
                    "vector": [final_label_centervectors1[int(kys.replace("cright_", ""))]],
                    "cts": [final_label_cts1[int(kys.replace("cright_", ""))]]}
        else:
            tmpxxt = f_final_label_centervectors[vlss]
            if "cleft_" in kys:
                tmpxxt_vector = tmpxxt["vector"]
                tmpxxt_vector.append(final_label_centervectors[int(kys.replace("cleft_", ""))])
                tmpxxt["vector"] = tmpxxt_vector

                tmpxxt_cts = tmpxxt["cts"]
                tmpxxt_cts.append(final_label_cts[int(kys.replace("cleft_", ""))])
                tmpxxt["cts"] = tmpxxt_cts
            else:
                tmpxxt_vector = tmpxxt["vector"]
                tmpxxt_vector.append(final_label_centervectors1[int(kys.replace("cright_", ""))])
                tmpxxt["vector"] = tmpxxt_vector

                tmpxxt_cts = tmpxxt["cts"]
                tmpxxt_cts.append(final_label_cts1[int(kys.replace("cright_", ""))])
                tmpxxt["cts"] = tmpxxt_cts
    nw_f_final_label_centervectors = {}
    for labelp, vectr_cts in f_final_label_centervectors.items():
        vt_lsts = vectr_cts["vector"]
        ct_lsts = vectr_cts["cts"]
        rates = [float(ct_lsi) / sum(ct_lsts) for ct_lsi in ct_lsts]
        sum_featp = np.zeros(vt_lsts[0].shape[0])
        for j in range(len(rates)):
            sum_featp = sum_featp + vt_lsts[j] * rates[j]
        nw_f_final_label_centervectors[labelp] = sum_featp

    return f_good_rets_dict, f_final_label_cts, nw_f_final_label_centervectors


def get_bad_rets_label_center(cid, bad_rets_lsts, id_cts):
    center_tmp = {}
    for i in range(len(bad_rets_lsts)):
        bad_rets_lst = bad_rets_lsts[i]
        feats = []
        ctss = []
        for idx in bad_rets_lst:
            feat = get_k_center(idx, cid)
            feats.append(feat)
            ctss.append(id_cts[idx])
        rates = [float(ctii) / sum(ctss) for ctii in ctss]
        sm_feat = np.zeros(feats[0].shape[0])
        for j in range(len(rates)):
            sm_feat = sm_feat + feats[j] * rates[j]
        center_tmp[i] = sm_feat
    return center_tmp


def write_result_to_demo(n_good_rets_dict, sid,after_proccess_rets,after_proccess_imgs):
    no_path_proccess_imgs=[imgpthi.split(r'/')[-1] for imgpthi in after_proccess_imgs]
    if not os.path.exists(r"try_demo_data/" + str(sid)):
        os.mkdir(r"try_demo_data/" + str(sid))
    for label, cid_ids_rets in tqdm(n_good_rets_dict.items()):
        if not os.path.exists(r"try_demo_data/" + sid + r"/" + str(label)):
            os.mkdir(r"try_demo_data/" + sid + r"/" + str(label))
        for cid, ids in cid_ids_rets.items():
            for id in ids:
                imgs = os.listdir(r"data/tstdata/" + cid + r"/" + id)
                for img in imgs:
                    if img in no_path_proccess_imgs:
                        continue
                    shutil.copy(r"data/tstdata/" + cid + r"/" + id + r"/" + img,
                                r"try_demo_data/" + sid + r"/" + str(label))
    for labelx,imgpths in after_proccess_rets.items():
        if not os.path.exists(r"try_demo_data/" + sid + r"/" + str(labelx)):
            os.mkdir(r"try_demo_data/" + sid + r"/" + str(labelx))
        for imgpth in imgpths:
            shutil.copy(imgpth,r"try_demo_data/" + sid + r"/" + str(labelx))



def writesid_result(cid_framid_ctid_bbox_dict,rets, start_n,after_proccess_rets,after_proccess_imgs):
    no_path_proccess_imgs = [imgpthi.split(r'/')[-1] for imgpthi in after_proccess_imgs]
    for label, tmps in rets.items():
        pid = label + start_n
        for cid, ids in tmps.items():
            framid_dict = get_framid_ctid_boundingbox_dict(cid)
            for id in ids:
                imgs = os.listdir(r"data/tstdata/" + str(cid) + "/" + str(id))
                for img in imgs:
                    if img in no_path_proccess_imgs:
                        continue
                    framid = int(img.split("_")[1])
                    ctid = int(img.split("_")[2])
                    xymnmxs = framid_dict[framid][ctid]
                    xmin, ymin, width, height = xymnmxs
                    f = open("track1.txt", "a", encoding="utf-8")
                    f_cid = cid.replace("c", "").replace("C", "")
                    f_cid = int(f_cid)
                    f.write(
                        str(f_cid) + " " + str(pid) + " " + str(framid) + " " + str(xmin) + " " + str(ymin) + " " + str(
                            width) + " " + str(height) + " 1 1\n")
                    f.close()
    for labelx,imgpths in after_proccess_rets.items():
        pidx = labelx + start_n
        for imgpth in imgpths:
            imgx=imgpth.split(r'/')[-1]
            framid = int(imgx.split("_")[1])
            ctid = int(imgx.split("_")[2])

            cid=imgpth.split(r"/")[2]
            framid_dictx = cid_framid_ctid_bbox_dict[cid]

            xymnmxs = framid_dictx[framid][ctid]
            xmin, ymin, width, height = xymnmxs
            f = open("track1.txt", "a", encoding="utf-8")
            f_cid = cid.replace("c", "").replace("C", "")
            f_cid = int(f_cid)
            f.write(
                str(f_cid) + " " + str(pidx) + " " + str(framid) + " " + str(xmin) + " " + str(ymin) + " " + str(
                    width) + " " + str(height) + " 1 1\n")
            f.close()



def get_init_add_cid_rets_dict(good_rets_dict, cid):
    rets = {}
    for label, ids in good_rets_dict.items():
        rets[label] = {cid: ids}
    return rets


def zc_pd_xj(cid_idset2, cids1, cid_idset1):
    for cid2, idsets2 in cid_idset2.items():
        if cid2 in cids1:
            ids1 = cid_idset1[cid2]
            id_maxmin_framid = get_id_maxmin_framid(cid2)
            is_jj = pd_two_idsets_is_jj(ids1, idsets2, id_maxmin_framid)
            if is_jj:
                return True
    return False


def combine_two_cid_ids_sets(center_rets_dicti, non_center_rets_dicti):
    rets = {}
    for cid1, ids1 in center_rets_dicti.items():
        if cid1 in non_center_rets_dicti:
            rets[cid1] = ids1 + non_center_rets_dicti[cid1]
        else:
            rets[cid1] = ids1
    for cid2, ids2 in non_center_rets_dicti.items():
        if cid2 not in rets:
            rets[cid2] = ids2
    return rets


def get_final_good_rets_clust(n_good_rets_dict, final_label_cts, final_label_centervectors, final_alpah):
    sum_cts = sum(list(final_label_cts.values()))
    nwrets = sorted(final_label_cts.items(), key=lambda x: x[1], reverse=True)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,',nwrets,'=======',sum_cts)
    center_labels = []
    for i in range(len(nwrets)):
        if nwrets[i][1] / float(sum_cts) >= 0.045:
            center_labels.append(nwrets[i][0])
    print(';;;;;;;;;;;;;;;;;;;;;;;;',len(center_labels),'????????????',center_labels)
    center_rets_dict = {}
    center_label_cts = {}
    center_label_centervectors = {}

    non_center_rets_dict = {}
    non_center_label_cts = {}
    non_center_label_centervectors = {}
    for label, cts in final_label_cts.items():
        if label in center_labels:
            center_rets_dict[label] = n_good_rets_dict[label]
            center_label_cts[label] = final_label_cts[label]
            center_label_centervectors[label] = final_label_centervectors[label]
        else:
            non_center_rets_dict[label] = n_good_rets_dict[label]
            non_center_label_cts[label] = final_label_cts[label]
            non_center_label_centervectors[label] = final_label_centervectors[label]

    if len(non_center_label_centervectors)>0:
        have_combines=[]
        for labelx, center_vectorx in non_center_label_centervectors.items():
            cid_idset1 = non_center_rets_dict[labelx]
            cids1 = list(cid_idset1.keys())
            tmpprets = {}
            for labely, center_vectory in center_label_centervectors.items():
                cid_idset2 = center_rets_dict[labely]
                tpiim = zc_pd_xj(cid_idset2, cids1, cid_idset1)
                print('==============',labelx,'+++++++',labely,':::',tpiim)
                if True:#not tpiim:
                    tmpxivalue = eucliDist(center_vectory, center_vectorx)
                    if tmpxivalue < final_alpah:
                        tmpprets[labely] = tmpxivalue
            nwrets = sorted(tmpprets.items(), key=lambda x: x[1], reverse=False)
            if len(nwrets) > 0:
                have_combines.append(labelx)
                mblabel = nwrets[0][0]
                center_rets_dict[mblabel] = combine_two_cid_ids_sets(center_rets_dict[mblabel],
                                                                     non_center_rets_dict[labelx])


                center_label_centervectors[mblabel] = center_label_centervectors[mblabel] * (
                        float(center_label_cts[mblabel]) / (center_label_cts[mblabel] + non_center_label_cts[labelx])) + \
                                                      non_center_label_centervectors[labelx] * (
                                                              float(non_center_label_cts[labelx]) / (
                                                              center_label_cts[mblabel] + non_center_label_cts[
                                                          labelx]))

                center_label_cts[mblabel] = center_label_cts[mblabel] + non_center_label_cts[labelx]
        rt_rets_dict={}
        rt_label_centervectors={}
        rt_label_cts={}
        mmm=0
        for labelz, center_vectorz in center_label_centervectors.items():
            rt_rets_dict[mmm]=center_rets_dict[labelz]
            rt_label_centervectors[mmm]=center_label_centervectors[labelz]
            rt_label_cts[mmm]=center_label_cts[labelz]
            mmm=mmm+1
        print('step4444444444444,..................',len(center_label_centervectors),'+++++++++++++++',list(center_label_centervectors.keys()),'.....',len(have_combines),',,,,,,,,,,,',len(non_center_label_centervectors))
        for labeln,noncenter_vectorn in non_center_label_centervectors.items():
            if labeln not in have_combines:
                rt_rets_dict[mmm] = non_center_rets_dict[labeln]
                rt_label_centervectors[mmm] = non_center_label_centervectors[labeln]
                rt_label_cts[mmm] = non_center_label_cts[labeln]
                mmm=mmm+1
        return rt_rets_dict,rt_label_cts,rt_label_centervectors
    else:
        return n_good_rets_dict, final_label_cts, final_label_centervectors

def combine_good_and_bat_rets(n_good_rets_dict,final_n_bad_rets_dict):
    final_total_rets_dict={}
    for labelg,cid_id_setsg in n_good_rets_dict.items():
        if labelg in final_n_bad_rets_dict:
            cid_id_setsb=final_n_bad_rets_dict[labelg]
            tmps={}
            for cidg,idsg in cid_id_setsg.items():
                if cidg in cid_id_setsb:
                    tmps[cidg]=list(set(idsg+cid_id_setsb[cidg]))
                else:
                    tmps[cidg]=idsg
            for cidb,idsb in cid_id_setsb.items():
                if cidb not in tmps:
                    tmps[cidb]=idsb
            final_total_rets_dict[labelg]=tmps
        else:
            final_total_rets_dict[labelg]=cid_id_setsg
    return final_total_rets_dict

def get_img_feat(imgpth):
    result = image_reid_person([imgpth])
    feat=result[0]['img_embedding'][0]
    return feat

def final_processdata(n_good_rets_dict,final_label_centervectors):
    other_imgs=[]
    other_feats=[]
    for label,cid_ids in n_good_rets_dict.items():
        for cid,ids in cid_ids.items():
            tmps={}
            for id in ids:
                imgs = os.listdir(r"data/tstdata/" + cid + r"/" + id)
                for img in imgs:
                    framid=int(img.split("_")[1])
                    if framid not in tmps:
                        tmps[framid]=[r"data/tstdata/" + cid + r"/" + id+r"/"+img]
                    else:
                        xxppi=tmps[framid]
                        xxppi.append(r"data/tstdata/" + cid + r"/" + id+r"/"+img)
                        tmps[framid]=xxppi
            for framidx,subimgs in tmps.items():
                if len(subimgs)>1:
                    label_centervector=final_label_centervectors[label]
                    subimgfeats=[get_img_feat(imgpth) for imgpth in subimgs]
                    dists=[eucliDist(label_centervector, subfeat) for subfeat in subimgfeats]
                    minindex=dists.index(min(dists))
                    for i in range(len(subimgs)):
                        if i!=minindex:
                            other_imgs.append(subimgs[i])
                            other_feats.append(subimgfeats[i])
    results={}
    for j in range(len(other_imgs)):
        disxts=[]
        featj = other_feats[j]
        labelxs=[]
        for labelx,center_featx in final_label_centervectors.items():
            disxts.append(eucliDist(featj,center_featx))
            labelxs.append(labelx)
        minindexj = disxts.index(min(disxts))
        bestlabel = labelxs[minindexj]
        if bestlabel not in results:
            results[bestlabel]=[other_imgs[j]]
        else:
            ttmmppjj=results[bestlabel]
            ttmmppjj.append(other_imgs[j])
            results[bestlabel]=ttmmppjj
    return results,other_imgs


def getsid_clust(sid, start_n):
    cids = []
    if sid == "S001":
        cids = ["c001", "c002", "c003", "c004", "c005", "c006", "c007"]
    elif sid == "S003":
        cids = ["c014", "c015", "c016", "c017", "c018", "c019"]
    elif sid == "S009":
        cids = ["c047", "c048", "c049", "c050", "c051", "c052"]
    elif sid == "S014":
        cids = ["c076", "c077", "c078", "c079", "c080", "c081"]
    elif sid == "S018":
        cids = ["c100", "c101", "c102", "c103", "c104", "c105"]
    elif sid == "S021":
        cids = ["c118", "c119", "c120", "c121", "c122", "c123"]
    elif sid == "S022":
        cids = ["c124", "c125", "c126", "c127", "c128", "c129"]
    if sid == "S001":
        alpha_value = 32.0
    elif sid == "S018":
        alpha_value = 24.0
    else:
        alpha_value = 32.0
    good_rets_dict, bad_rets_lsts, final_label_cts, final_label_centervectors = new_clust_step2.good_rets_main(cids[0],
                                                                                                               sid)
    n_good_rets_dict = get_init_add_cid_rets_dict(good_rets_dict, cids[0])
    for i in range(1, len(cids)):
        id_maxmin_framid = get_id_maxmin_framid(cids[i])
        print("-----*****************--------", i, "++++*******************+++", list(n_good_rets_dict.keys()),
              '+++***********+++', final_label_cts)
        n_good_rets_dict, final_label_cts, final_label_centervectors = get_two_different_cid_clust(id_maxmin_framid,
                                                                                                   n_good_rets_dict,
                                                                                                   final_label_cts,
                                                                                                   final_label_centervectors,
                                                                                                   cids[i], sid,
                                                                                                   alpha=alpha_value)  # 32.0
        print("------------------------", i, "+++++++++++++++++++++++++++++++", list(n_good_rets_dict.keys()),
              '++++++++++++', final_label_cts)
    print("-@@@@@@@@@@@@@@-", i, "++++++@@@@@@@@@@@@@@@+++", list(n_good_rets_dict.keys()), '+++@@@@@@@+++',
          final_label_cts)
    if sid == "S001":
        final_alpah = 32.0
    elif sid == "S018":
        final_alpah = 32.0
    else:
        final_alpah = 32.0
    n_good_rets_dict, final_label_cts, final_label_centervectors=get_final_good_rets_clust(n_good_rets_dict, final_label_cts, final_label_centervectors, final_alpah=final_alpah)
    print("-----=============--", i, "++++============+++++++", list(n_good_rets_dict.keys()), '+++========+++',
          final_label_cts)

    #write_result_to_demo(n_good_rets_dict, sid)
    #writesid_result(n_good_rets_dict,start_n)

    final_n_bad_rets_dict = {}

    for cid in cids:
        id_cts = get_id_ct(cid)
        _, bad_rets_idss, _, _ = new_clust_step2.good_rets_main(cid, sid)
        bad_label_centers = get_bad_rets_label_center(cid, bad_rets_idss, id_cts)
        for xlabel, bad_label_center in bad_label_centers.items():
            ids = bad_rets_idss[xlabel]
            tmpiis = {}
            for rlabel, rcenter in final_label_centervectors.items():
                tmpiis[rlabel] = eucliDist(bad_label_center, rcenter)
            tmpiis_lst = sorted(tmpiis.items(), key=lambda x: x[1], reverse=False)
            f_label = tmpiis_lst[0][0]
            if f_label not in final_n_bad_rets_dict:
                final_n_bad_rets_dict[f_label] = {cid: ids}
            else:
                tmpyyiis = final_n_bad_rets_dict[f_label]
                if cid not in tmpyyiis:
                    tmpyyiis[cid] = ids
                else:
                    tmpyyiisxx = tmpyyiis[cid]
                    tmpyyiisxx = list(set(tmpyyiisxx + ids))
                    tmpyyiis[cid] = tmpyyiisxx
                final_n_bad_rets_dict[f_label] = tmpyyiis



    final_total_rets_dict=combine_good_and_bat_rets(n_good_rets_dict,final_n_bad_rets_dict)

    after_proccess_rets, after_proccess_imgs = final_processdata(final_total_rets_dict, final_label_centervectors)

    write_result_to_demo(final_total_rets_dict, sid,after_proccess_rets,after_proccess_imgs)
    

    #cid_framid_ctid_bbox_dict={}
    #for cid in cids:
    #    cid_fid_ct_tmps=get_framid_ctid_boundingbox_dict(cid)
    #    cid_framid_ctid_bbox_dict[cid]=cid_fid_ct_tmps
    #writesid_result(cid_framid_ctid_bbox_dict,final_total_rets_dict,start_n,after_proccess_rets,after_proccess_imgs)



if __name__ == "__main__":
    sids = ["S001", "S003", "S009", "S014", "S018", "S021", "S022"]
    start_ns = [10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]
    for i in range(len(sids)):
        sid = sids[i]
        start_n = start_ns[i]
        getsid_clust(sid, start_n)


