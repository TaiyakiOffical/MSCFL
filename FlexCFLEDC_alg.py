# 主函数
import torch
import datetime
import time
import torch.nn as nn
from models import *
import numpy as np
import random
from sklearn.cluster import KMeans
import logging
from training import train_net
from sklearn.metrics.pairwise import cosine_similarity
from miscellaneous import get_args, print2log
from data_process import partition_data
from client import Client_Sub_S
import torch.nn.functional as F
import copy
import weightsPrune as weightsPrune
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD


def check_correct_cluster(idx, client_selection, correct_cluster_group):
    if all(elem in client_selection for elem in correct_cluster_group[idx]):
        for i in range(len(correct_cluster_group)):
            if correct_cluster_group[i] != correct_cluster_group[idx] and all(
                elem in client_selection for elem in correct_cluster_group[i]
            ):
                return False
        return True
    else:
        return False


def flatten_state_dict(state_dicts):
    """将状态字典中的参数拉平成向量"""
    vectors = []
    for state_dict in state_dicts:
        vector = []
        for param in state_dict.values():
            vector.extend(param.cpu().numpy().flatten())
        vectors.append(vector)
    return np.array(vectors)


def kmeans_clustering(model_state_dicts, num_clusters):
    # 执行k-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(model_state_dicts)

    return kmeans.labels_


def kmean_clustering(data):
    # 将一维数组转换为二维数组
    data_2d = data.reshape(-1, 1)

    # 使用K均值聚类算法将数据分成两类
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data_2d)

    # 获取每个数据点所属的簇
    labels = kmeans.labels_
    print(labels)
    print(data)
    # 将数据分成两类
    cluster1 = data[labels == 0]
    cluster2 = data[labels == 1]
    if cluster1[0] > cluster2[0]:
        indices = np.where(labels == 0)[0]
        return indices
    else:
        indices = np.where(labels == 1)[0]
        return indices


# def flatten_score(score):
#     flatten_score = np.concatenate(score)
#     return flatten_score


def flatten_score(mask):
    mfc_cal = []
    for tensor in mask:
        flatten_param = tensor.flatten()
        mfc_cal.append(flatten_param)
    flatten_mfc_cal = np.concatenate(mfc_cal)
    return flatten_mfc_cal


def flatten_mask(mask):
    mfc_cal = []
    for tensor in mask:
        flatten_param = tensor.flatten()
        mfc_cal.append(flatten_param)
        flatten_mfc_cal = np.concatenate(mfc_cal)
    return flatten_mfc_cal


def run_task(args):
    print2log("---------------Start Main Task--------------------")
    total_clustering_time = 0
    cluster_switch = True
    client_selection_matrix = []
    clusters_num = int(args.class_num / args.target_kinds)
    amount_in_cluster = int(args.n_clients / clusters_num)
    prune_ratio = np.random.normal(
        args.prune_ratio_mean, args.prune_ratio_std, args.n_clients
    )
    prune_ratio = np.clip(prune_ratio, 0.2, 0.7)
    print2log(f"prune_ratio:{prune_ratio}")

    model_pool, net_glob, data_size = init_net(args)
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
    w_servers_first = []
    correct_cluster_group = [
        [i, i + 1] for i in range(0, args.n_clients, 2) for _ in range(2)
    ]
    recluster = [False for _ in range(args.n_clients)]
    while current_round != 0:
        correct_cluster_number = 0
        mask_total_ch = []
        mask_total_fc = []
        masks_ch = []
        masks_fc = []
        current_round -= 1
        # 客户端开始训练
        for idx in range(args.n_clients):
            print2log(f"client{idx} starts to train")
            #
            if current_round == args.total_rounds - 1:
                clients[idx].train_firstRound(
                    prune_ratio[idx],
                    int(prune_ratio[idx] * 100),
                    copy.deepcopy(net_glob),
                    True,
                )
                # print(f"client{idx} score.shape:{np.array(clients[idx].score_fc).shape}")
            else:
                if (
                    args.swap_label == True
                    and current_round % 5 == 0
                    and random.random() < 0.3
                    and idx % 2 == 0
                    and idx < 8
                ):
                    if args.partition == "quantity_based_LDS":
                        numbers = [x for x in range(idx + 2, args.n_clients, 2)]
                    else:
                        numbers = [x for x in range(idx + 1, args.n_clients, 2)]
                    idx2 = random.choice(numbers)
                    temp_ldr_train = copy.deepcopy(clients[idx].ldr_train)
                    temp_ldr_test = copy.deepcopy(clients[idx].ldr_test)
                    clients[idx].ldr_train = copy.deepcopy(clients[idx2].ldr_train)
                    clients[idx].ldr_test = copy.deepcopy(clients[idx2].ldr_test)
                    clients[idx2].ldr_train = temp_ldr_train
                    clients[idx2].ldr_test = temp_ldr_test
                    print2log(f"client{idx} has swapped the dataset with client{idx2}")
                    if args.partition == "quantity_based_LDS":
                        recluster[idx] = True
                        recluster[idx2] = True
                if recluster[idx]:
                    clients[idx].net = copy.deepcopy(net_glob)
                    clients[idx].mask_ch = copy.deepcopy(mask_ch_init)
                    clients[idx].mask_fc = copy.deepcopy(mask_fc_init)
                    clients[idx].normal_train()
                    cosine_similarity_list = []
                    params1 = torch.cat(
                        [param.view(-1) for param in clients[i].net.parameters()]
                    )
                    for i in range(args.n_clients):
                        new_model = copy.deepcopy(net_glob)
                        new_model.load_state_dict(w_servers_first[i])
                        params2 = torch.cat(
                            [param.view(-1) for param in new_model.parameters()]
                        )
                        model_cosine_similarity = F.cosine_similarity(
                            params1.unsqueeze(0), params2.unsqueeze(0)
                        ).item()
                        cosine_similarity_list.append(model_cosine_similarity)
                    chosen_idx = cosine_similarity_list.index(
                        max(cosine_similarity_list)
                    )
                    for i, list in enumerate(client_selection_matrix):
                        if idx in list:
                            client_selection_matrix[i] = np.delete(
                                client_selection_matrix[i],
                                np.where(client_selection_matrix[i] == idx),
                            )
                    client_selection_matrix[idx] = copy.deepcopy(
                        client_selection_matrix[chosen_idx]
                    )
                    temp = copy.deepcopy(client_selection_matrix[chosen_idx])
                    for i, list in enumerate(client_selection_matrix):
                        if np.array_equal(list, temp):
                            client_selection_matrix[i] = np.append(
                                client_selection_matrix[i], idx
                            )
                    print2log(f"client_selection_matrix:{client_selection_matrix}")
                    clients[idx].train_firstRound(
                        prune_ratio[idx],
                        int(prune_ratio[idx] * 100),
                        copy.deepcopy(net_glob),
                        True,
                    )
                    recluster[idx] = False
                else:
                    clients[idx].train(
                        prune_ratio[idx],
                        int(prune_ratio[idx] * 100),
                        copy.deepcopy(net_glob),
                        True,
                    )
            print2log(f"client{idx} finishes training")

        for idx in range(args.n_clients):
            if data_size != None:
                masks_ch.append(copy.deepcopy(clients[idx].get_mask_ch()))
            masks_fc.append(copy.deepcopy(clients[idx].get_mask_fc()))

        # 确定所有客户端的所选聚合客户端的矩阵
        if cluster_switch == True:
            para_lists = np.array(
                [
                    np.concatenate(clients[idx].para_list).tolist()
                    for idx in range(args.n_clients)
                ]
            )
            for i, row in enumerate(para_lists):
                print(f"Shape of row {i+1}: {np.shape(row)}")
            begin = time.time()
            svd = TruncatedSVD(n_components=clusters_num)
            decomposed_value = svd.fit_transform(para_lists.T)
            decomposed_matrix = cosine_similarity(
                para_lists, decomposed_value.T
            )  # shape=(n_clients, n_clients)
            labels = KMeans(clusters_num, random_state=0).fit(decomposed_matrix).labels_
            end = time.time()
            for idx in range(args.n_clients):
                client_selection = np.where(labels == labels[idx])[0]
                client_selection_matrix.append(client_selection)
                if args.check_cluster_acc and check_correct_cluster(
                    idx, client_selection.tolist(), correct_cluster_group
                ):
                    correct_cluster_number += 1
            print2log(f"client_selection_matrix:{client_selection_matrix}")
            total_clustering_time += end - begin
            cluster_switch = False

        # p2p聚合模型
        w_servers = []
        loss_list = []
        acc_list = []
        for idx in range(args.n_clients):
            w_clients = []
            if data_size != None:
                temp_masks_ch = []
            temp_masks_fc = []
            for index in client_selection_matrix[idx]:
                index = int(index)
                w_clients.append(clients[index].get_state_dict())
                if data_size != None:
                    temp_masks_ch.append(masks_ch[index])
                temp_masks_fc.append(masks_fc[index])
            if data_size != None:
                w_server = clients[idx].Sub_FedAVG_S(
                    copy.deepcopy(net_glob.state_dict()),
                    w_clients,
                    temp_masks_ch,
                    temp_masks_fc,
                    copy.deepcopy(net_glob),
                    args.ks,
                    args.in_ch,
                )
            else:
                w_server = clients[idx].Sub_FedAVG_U(
                    copy.deepcopy(net_glob.state_dict()), w_clients, temp_masks_fc
                )
            w_servers.append(w_server)

        print2log(f"cluster accuracy:{correct_cluster_number/args.n_clients}")
        if current_round == args.total_rounds - 1:
            w_servers_first = copy.deepcopy(w_servers)
        # 用聚合模型代替现有模型
        for idx in range(args.n_clients):
            clients[idx].net = copy.deepcopy(net_glob)
            clients[idx].net.load_state_dict(w_servers[idx])
            test_loss, accuracy = clients[idx].eval_test(clients[idx].net)
            print2log(f"test_loss:{test_loss},accuracy:{accuracy}")
            loss_list.append(test_loss)
            acc_list.append(accuracy)
        print2log(f"acc_mean:{sum(acc_list)/args.n_clients}")
        print2log(f"loss_mean:{sum(loss_list)/args.n_clients}")
    print2log(f"total_clustering_time:{total_clustering_time}")


def set_log(args):
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
    if args.partition == "quantity_based_LDS":
        log_file_path = os.path.join(args.logdir, f"FlexCFL-label-{start_time}.log")
    else:
        log_file_path = os.path.join(args.logdir, f"FlexCFL-feature-{start_time}.log")
    if not os.path.exists(args.logdir):
        # 如果路径不存在，则创建路径
        os.makedirs(args.logdir)
    log_mode = logging.INFO
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=log_mode,
        filemode="a",
    )


def Main():
    args = get_args()
    set_log(args)
    args.cfg_prune = (
        [2, "M", 5, "M"]
        if args.dataset != "cifar100"
        else [1, "M", 1, "M", 1, "M", 1, "M"]
    )
    run_task(args)
    return


if __name__ == "__main__":
    Main()
