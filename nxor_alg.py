# 主函数
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F

from models import *
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import logging
from training import train_net
from miscellaneous import get_args, print2log
from data_process import partition_data
from client import Client_Sub_S
import copy
import xzf.MSCFL.weightsPrune as weightsPrune
import os


def hamming_distance(x, y):
    return np.sum(x != y)


def kmeans_hamming(X, k, max_iter=300):
    # 初始化簇中心：随机选择 k 个数据点作为簇中心
    centers = X[np.random.choice(X.shape[0], k, replace=False)]

    # 记录每个点的簇标签
    labels = np.zeros(X.shape[0])

    for _ in range(max_iter):
        # 步骤 1：分配簇标签
        for i in range(X.shape[0]):
            distances = np.array([hamming_distance(X[i], center) for center in centers])
            labels[i] = np.argmin(distances)

        # 步骤 2：更新簇中心
        new_centers = np.zeros_like(centers)
        for j in range(k):
            # 选择每个簇中出现最多的 0 和 1 来更新簇中心
            cluster_points = X[labels == j]
            new_centers[j] = np.array(
                [
                    np.round(np.mean(cluster_points[:, feature]))
                    for feature in range(X.shape[1])
                ]
            )

        # 检查是否收敛，如果簇中心没有变化则提前结束
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels


def prune_twice(score1, score2):
    for step in range(len(score1)):
        if len(score1[step][score1[step] != 0]) < len(score2[step][score2[step] != 0]):
            percent = (
                100 - len(score1[step][score1[step] != 0]) / len(score1[step]) * 100
            )
            print(f"prune rate1:{percent}")
            percent_value = np.percentile(score2[step], percent)
            score2[step][score2[step] < percent_value] = 0
            print(
                f"prune rate2:{len(score2[step][score2[step] < percent_value])/len(score2[step])}"
            )
        else:
            percent = (
                100 - len(score2[step][score2[step] != 0]) / len(score2[step]) * 100
            )
            print(f"prune rate1:{percent}")
            percent_value = np.percentile(score1[step], percent)
            score1[step][score1[step] < percent_value] = 0
            print(
                f"prune rate2:{len(score1[step][score1[step] < percent_value])/len(score1[step])}"
            )
    return score1, score2


def kmean_clustering(data, clustering_level=3):
    # 将一维数组转换为二维数组
    data_2d = data.reshape(-1, 1)

    # 使用K均值聚类算法将数据分成两类
    kmeans = KMeans(n_clusters=clustering_level)
    kmeans.fit(data_2d)

    # 获取每个数据点所属的簇
    labels = kmeans.labels_
    print(labels)
    print(data)
    max_cluster = [0]
    index = -1
    for i in range(clustering_level):
        cluster = data[labels == i]
        if cluster[0] > max_cluster[0]:
            max_cluster = cluster
            index = i
    indices = np.where(labels == index)[0]
    return indices


def flatten_score(score):
    flatten_score = np.concatenate(score)
    return flatten_score


def flatten_mask(mask):
    mfc_cal = []
    for tensor in mask:
        flatten_param = tensor.flatten()
        mfc_cal.append(flatten_param)
        flatten_mfc_cal = np.concatenate(mfc_cal)
    return flatten_mfc_cal


def run_task(args):
    print2log("---------------Start Main Task--------------------")
    clusters_num = int(args.class_num / args.target_kinds)
    amount_in_cluster = int(args.n_clients / clusters_num)
    prune_ratio = np.random.normal(
        args.prune_ratio_mean, args.prune_ratio_std, args.n_clients
    )
    prune_ratio = np.clip(prune_ratio, 0.1, 0.8)
    print2log(f"prune_ratio:{prune_ratio}")

    model_pool, net_glob, data_size = init_net(args)
    cfg_before = model_pool[0].cfg
    party_dataloader_train, party_dataloader_test, party_dataSize_dict = partition_data(
        args.dataset,
        args.datadir,
        args.partition,
        args.n_clients,
        args.batch_size,
        noise=args.noise,
        poisoning_type=args.poisoning_type,
        target_kinds=args.target_kinds,
        seed=args.seed,
        beta=args.beta,
        alpha=args.partition_alpha,
    )
    print2log("---------------Data Partition Complete--------------------")
    mask_ch_init = weightsPrune.make_init_mask_ch(net_glob)
    mask_fc_init = weightsPrune.make_init_mask_fc(net_glob)
    print2log("---------------Mask Init Complete--------------------")
    clients = []
    for idx in range(args.n_clients):
        clients.append(
            Client_Sub_S(
                idx,
                model_pool[idx],
                args.batch_size,
                args.epochs,
                args.first_train_ep,
                args.first_train_ep_ratio,
                args.lr,
                args.momentum,
                args.device,
                copy.deepcopy(mask_ch_init),
                copy.deepcopy(mask_fc_init),
                args.cfg_prune,
                args.in_ch,
                args.ks,
                args,
                party_dataloader_train[idx],
                party_dataloader_test[idx],
                data_size,
            )
        )
    print2log("---------------Clients Init Complete,Start to Train--------------------")
    current_round = args.total_rounds
    while current_round != 0:
        current_round -= 1
        mask_total_ch = []
        mask_total_fc = []
        mask_total = []
        masks_ch = []
        masks_fc = []
        # 客户端开始训练
        for idx in range(args.n_clients):
            print2log(f"client{idx} starts to train")
            if current_round == args.total_rounds - 1:
                clients[idx].train_firstRound(
                    prune_ratio[idx],
                    int(prune_ratio[idx] * 100),
                    copy.deepcopy(net_glob),
                    True,
                )
                # print(f"client{idx} score.shape:{np.array(clients[idx].score_fc).shape}")
            else:
                clients[idx].train(
                    prune_ratio[idx],
                    int(prune_ratio[idx] * 100),
                    copy.deepcopy(net_glob),
                    True,
                )
            print2log(f"client{idx} finishes training")

        for idx in range(args.n_clients):
            mask_total_ch.append(clients[idx].mch.numpy())
            mask_total_fc.append(flatten_mask(clients[idx].mask_fc))
            mask_total.append(
                np.concatenate(
                    (clients[idx].mch.numpy(), flatten_mask(clients[idx].mask_fc))
                )
            )
            masks_ch.append(copy.deepcopy(clients[idx].get_mask_ch()))
            masks_fc.append(copy.deepcopy(clients[idx].get_mask_fc()))

        # 确定分数矩阵
        score_matrix_total = np.zeros((args.n_clients, args.n_clients))

        for idx1 in range(args.n_clients):
            for idx2 in range(args.n_clients):
                if idx1 != idx2:
                    client_idx1_score_ch = flatten_score(
                        copy.deepcopy(clients[idx1].score_ch)
                    )
                    client_idx1_score_fc = flatten_score(
                        copy.deepcopy(clients[idx1].score_fc)
                    )
                    client_idx2_score_ch = flatten_score(
                        copy.deepcopy(clients[idx2].score_ch)
                    )
                    client_idx2_score_fc = flatten_score(
                        copy.deepcopy(clients[idx2].score_fc)
                    )
                    mask_total_ch[idx1] = [
                        1 if client_idx1_score_ch[i] != 0 else 0
                        for i in range(len(client_idx1_score_ch))
                    ]
                    mask_total_ch[idx2] = [
                        1 if client_idx2_score_ch[i] != 0 else 0
                        for i in range(len(client_idx2_score_ch))
                    ]
                    mask_total_fc[idx1] = [
                        1 if client_idx1_score_fc[i] != 0 else 0
                        for i in range(len(client_idx1_score_fc))
                    ]
                    mask_total_fc[idx2] = [
                        1 if client_idx2_score_fc[i] != 0 else 0
                        for i in range(len(client_idx2_score_fc))
                    ]
                    temp = np.array(
                        [
                            1 if mask_total_ch[idx1][i] == mask_total_ch[idx2][i] else 0
                            for i in range(len(mask_total_ch[idx1]))
                        ]
                    )
                    score = sum(temp)
                    temp = np.array(
                        [
                            1 if mask_total_fc[idx1][i] == mask_total_fc[idx2][i] else 0
                            for i in range(len(mask_total_fc[idx1]))
                        ]
                    )
                    score += sum(temp)
                    score_matrix_total[idx1][idx2] = score
        print2log(f"score_matrix_total:{score_matrix_total}")
        # 确定所有客户端的所选聚合客户端的矩阵
        labels = KMeans(n_clusters=clusters_num, random_state=0).fit(mask_total).labels_
        client_selection_matrix = []
        for idx in range(args.n_clients):
            client_selection = np.where(labels == labels[idx])[0]
            client_selection_matrix.append(client_selection)
        print2log(f"client_selection_matrix:{client_selection_matrix}")

        # p2p聚合模型
        w_servers = []
        for idx in range(args.n_clients):
            w_clients = []
            temp_masks_ch = []
            temp_masks_fc = []
            for index in client_selection_matrix[idx]:
                index = int(index)
                w_clients.append(clients[index].get_state_dict())
                temp_masks_ch.append(masks_ch[index])
                temp_masks_fc.append(masks_fc[index])
            w_server = clients[idx].Sub_FedAVG_S(
                copy.deepcopy(net_glob.state_dict()),
                w_clients,
                temp_masks_ch,
                temp_masks_fc,
                copy.deepcopy(net_glob),
                args.ks,
                args.in_ch,
            )
            w_servers.append(w_server)
            test_loss, accuracy = clients[idx].eval_test(clients[idx].net)
            print(f"test_loss:{test_loss},accuracy:{accuracy}")

        # 用聚合模型代替现有模型
        for idx in range(args.n_clients):
            clients[idx].net = copy.deepcopy(net_glob)
            clients[idx].net.load_state_dict(w_servers[idx])


def set_log():
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
    log_file_path = os.path.join(".", "logs", f"nxor_{start_time}.log")
    log_mode = logging.INFO
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=log_mode,
        filemode="a",
    )


def Main():
    np.set_printoptions(threshold=np.inf)
    set_log()
    args = get_args()
    args.cfg_prune = [3, "M", 8, "M"]
    run_task(args)
    return


if __name__ == "__main__":
    Main()
