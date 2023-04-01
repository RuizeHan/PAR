import numpy as np
import nmslib
import json
import multiprocessing
import os
from .utils import log

def load_feats(fn, feature_dim=256):
    return np.fromfile(fn, dtype=np.float32).reshape(-1, feature_dim)

def dump2json(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, set) or isinstance(obj, np.ndarray):
            return list(obj)
        else:
            raise TypeError(
            "Unserializable object {} of type {}".format(obj, type(obj)))

    with open(ofn, 'w') as of:
        json.dump(data, of, default=default)

def knn_nmslib(feats, k, space='cosinesimil'):
    index = nmslib.init(method='hnsw', space=space)
    index.addDataPointBatch(feats)
    index.createIndex({'post': 2}, print_progress=True)
    neighbours = index.knnQueryBatch(feats, k=k, num_threads=multiprocessing.cpu_count())
    return neighbours

def knn_faiss(feats, k):
    import torch
    import faiss
    import pdb
    N, dim = feats.shape
    res = faiss.StandardGpuResources()
    feats /= np.linalg.norm(feats).reshape(-1, 1)
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(feats)
    D, I = index.search(feats, k + 1)
    pdb.set_trace()

def get_hist(topk):
    hist = {}
    for t in topk:
        for i in t:
            if i not in hist.keys():
                hist[i] = 1
            else:
                hist[i] += 1
    return hist

def fill_array(array, fill, length):
    assert length >= array.shape[0], "Cannot fill"
    if length == array.shape[0]:
        return array
    array2 = fill * np.ones((length), dtype=array.dtype)
    array2[:array.shape[0]] = array
    return array2

def create_knn(args, data_name):
    members = [args.base] + args.committee
    output_dir = 'data/{}/knn/'.format(data_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open("data/{}/list.txt".format(data_name), 'r') as f:
        fns = f.readlines()
    args.total_num = len(fns)

    # check feature files exist
    for m in members:
        if not os.path.isfile('data/{}/features/{}.bin'.format(data_name, m)):
            raise Exception('Feature file not exist: data/{}/features/{}.bin'.format(data_name, m))

    # create knn files with nmslib
    log("KNN Processing for: {}".format(data_name))
    for m in members:
        fn = "{}/{}_k{}.npz".format(output_dir, m, args.k)
        if not os.path.isfile(fn):
            feats = load_feats('data/{}/features/{}.bin'.format(data_name, m), args.feat_dim)
            assert feats.shape[0] == args.total_num, "Feature length of [{}] not consistent with list file, {} vs {}".format(m, feats.shape[0], args.total_num)
            log("\n\tSearch KNN for {}".format(m))
            if not hasattr(args, 'knn_method') or args.knn_method == 'nmslib':
                neighbours = knn_nmslib(feats, args.k)
            elif args.knn_method == 'faiss':
                neighbours = knn_faiss(feats, args.k)
            else:
                raise Exception("No such knn method: {}".format(args.knn_method))

            length = np.array([len(n[0]) for n in neighbours])
            tofill = np.where(length < args.k)[0]
            for idx in tofill:
                neighbours[idx] = [fill_array(neighbours[idx][0], -1, args.k), fill_array(neighbours[idx][1], -1., args.k)]
            knn_idx = np.array([n[0] for n in neighbours])
            knn_dist = np.array([n[1] for n in neighbours])
            np.savez(fn, idx=knn_idx, dist=knn_dist)
            log("\n")
        else:
            log("\tKNN file already exists: {}".format(fn))
