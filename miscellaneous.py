import argparse
import logging
import multiprocessing
import sys

import pandas as pd

import numpy as np
import torch
import os

from torch import optim
from torch.autograd import Variable


def print2log(message, level="INFO"):
    # 用于tgfl_discrete.py
    message = str(message)
    # print(message)
    if level == "INFO":
        logging.info(message)
    elif level == "DEBUG":
        logging.debug(message)
    else:
        pass


def __set_logger():
    # log_file_path = os.path.join("./logs/", 'dagfl_logs-{0}.log'.format(datetime.datetime.now().strftime("%H-%M")))
    log_file_path = os.path.join(".", "logs", "dagfl_logs.log")
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="a",
    )


def add_noise_to_local_model(net, std_devi, device="cpu"):
    temp_model_dict = net.state_dict()
    for key in temp_model_dict.keys():
        shape = temp_model_dict[key].size()
        ran = torch.normal(0, std_devi, shape)
        temp_model_dict[key] = temp_model_dict[key] + Variable(ran).to(device)

    net.load_state_dict(temp_model_dict)


def multi_krum(nets_para, parties_count, malicious_count, m=1):
    distances = {i: {} for i in range(parties_count)}
    nets_scores = {i: 0.0 for i in range(parties_count)}

    # # 需满足前提条件
    # assert parties_count > 2 * malicious_count + 2, ('parties_count > 2 * malicious_count + 2', parties_count, malicious_count)

    # 计算距离
    for i in range(parties_count):
        for j in range(i):
            distances[i][j] = distances[j][i] = __calculate_nets_distance(
                nets_para[i], nets_para[j]
            )

    # 计算每个net的得分
    for i in range(parties_count):
        scores = sorted(distances[i].values())
        nets_scores[i] = sum(scores[: parties_count - malicious_count - 1])
    nets_scores_items = sorted(nets_scores.items(), key=lambda x: x[1])

    # print(nets_scores_items)

    return [item[0] for item in nets_scores_items][:m]


def __calculate_nets_distance(net1_para, net2_para):
    parameters1, parameters2 = np.array([]), np.array([])

    for k1, k2 in zip(net1_para.keys(), net2_para.keys()):
        parameters1 = np.append(parameters1, net1_para[k1].cpu().numpy())
        parameters2 = np.append(parameters2, net2_para[k2].cpu().numpy())

    return np.linalg.norm(parameters1 - parameters2)


def get_size(obj, seen=None):
    # From
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_args():
    parser = argparse.ArgumentParser()

    # machine learning related
    parser.add_argument("--n_clients", type=int, default=10, help="客户端的数量")
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="cifar10, mnist, cifar100"
    )
    parser.add_argument("--class_num", type=int, default=10, help="该数据集有多少标签")
    parser.add_argument(
        "--partition",
        type=str,
        default="quantity_based_LDS",
        help="数据划分方法（iid和quantity_based_LDS,feature_imbalance_divide_four,feature_imbalance_divide_two注意后者不能与label-swapped投毒攻击同时存在）",
    )
    parser.add_argument(
        "--partition_alpha", type=float, default=1, help="noniid数据集的不平衡度"
    )
    parser.add_argument(
        "--partition_alpha2", type=float, default=1, help="noniid数据集的不平衡度"
    )
    parser.add_argument("--cluster_method", type=str, default="agglomerative")
    parser.add_argument("--prune_method", type=str, default="weightsPrune")

    parser.add_argument(
        "--target_kinds", type=int, default=2, help="每个节点或者客户端拥有多少个标签"
    )
    parser.add_argument("--model", type=str, default="lenet5", help="model name")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="number of local epochs")
    parser.add_argument(
        "--first_train_ep", type=int, default=10, help="number of first local epochs"
    )
    parser.add_argument(
        "--first_train_ep_ratio",
        type=float,
        default=1,
        help="the epoch before the prune ratio",
    )
    parser.add_argument(
        "--swap_label", type=bool, default=False, help="是否随机交换标签"
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        default=0.0,
        help="Dropout probability. Default=0.0",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    parser.add_argument(
        "--rho", type=float, default=0, help="Parameter controlling the momentum SGD"
    )
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument(
        "--reg", type=float, default=1e-5, help="L2 regularization strength"
    )
    parser.add_argument(
        "--poisoning_type",
        type=str,
        default="noise",
        help="模拟投毒攻击的方法，noise或者label-swapped",
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="seed for running dagfl_discrete"
    )
    parser.add_argument(
        "--noise", type=float, default=0, help="how much noise we add to some party"
    )
    parser.add_argument("--beta", type=int, default=1, help="狄利克雷分布参数")
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--ks", type=int, default=5, help="kernel size to use for convolutions"
    )
    parser.add_argument(
        "--in_ch", type=int, default=1, help="input channels of the first conv layer"
    )
    parser.add_argument(
        "--hamming_distance_alpha",
        type=float,
        default=1.1,
        help="input channels of the first conv layer",
    )
    parser.add_argument(
        "--check_cluster_acc",
        type=bool,
        default=False,
        help="input channels of the first conv layer",
    )

    parser.add_argument(
        "--datadir", type=str, required=False, default="./data/", help="Data directory"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "--log_file_name", type=str, default=None, help="The log file name"
    )
    parser.add_argument("--log_mode", type=str, default="DEBUG")
    parser.add_argument("--total_rounds", type=int, default=35)

    parser.add_argument("--prune_ratio_mean", type=float, default=0.35)
    parser.add_argument("--prune_ratio_std", type=float, default=0.3)

    parser.add_argument(
        "--pruning_percent_ch",
        type=float,
        default=0.25,
        help="Pruning percent for channels (0-1)",
    )
    parser.add_argument(
        "--pruning_percent_fc",
        type=float,
        default=40,
        help="Pruning percent for fully connected layers (0-100)",
    )
    # parser.add_argument('--use_projection_head', type=bool, default=False,
    #                     help='whether add an additional header to model or not (see MOON)')
    # parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    # parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    # parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    args = parser.parse_args()

    return args
