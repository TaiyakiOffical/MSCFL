import numpy as np
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial import distance
from miscellaneous import get_args, print2log

# from xzf.MSCFL.weightsPrune import *


def dist_masks(m1, m2):
    """
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it

    :param m1: pruning mask 1
    :param m2: pruning mask 2

    :return average hamming distance of two pruning masks:
    """
    temp_dist = []
    for step in range(len(m1)):
        # 1 - float(m1[step].reshape([-1]) == m2[step].reshape([-1])) / len(m2[step].reshape([-1]))
        temp_dist.append(
            distance.hamming(m1[step].reshape([-1]), m2[step].reshape([-1]))
        )
    dist = np.mean(temp_dist)
    return dist


args = get_args()
if args.prune_method == "weightsPrune":
    from weightsPrune import *
if args.prune_method == "filterPrune":
    from filterPrune import *
if args.prune_method == "fedDsePrune":
    from fedDsePrune import *


class Client_Sub_S(object):
    def __init__(
        self,
        name,
        model,
        local_bs,
        local_ep,
        first_train_ep,
        first_train_ep_ratio,
        lr,
        momentum,
        device,
        mask_ch,
        mask_fc,
        cfg_prune,
        in_ch,
        ks,
        args,
        ldr_train,
        ldr_test,
        ds,
        score_before_prune=False,
    ):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.first_train_ep = first_train_ep  # 第一次训练的总共轮数
        self.first_train_before_prune_ep_ratio = (
            first_train_ep_ratio  # 第一次训练裁剪前后的轮数比例
        )
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = ldr_train
        self.ldr_test = ldr_test
        self.mask_ch = mask_ch
        self.mask_fc = mask_fc
        self.cfg_prune = cfg_prune
        self.in_ch = in_ch
        self.ks = ks
        self.args = args
        self.acc_best = 0
        self.count = 0
        self.pruned_total = 0
        self.pruned_ch = 0
        self.pruned_fc = 0
        self.pruned_ch_rtonet = 0
        self.pruned_fc_rtonet = 0
        self.score_fc = None
        self.score_ch = None
        self.save_best = True
        self.ds = ds
        self.para_list = None
        self.score_before_prune = score_before_prune

    def train_firstRound(self, percent_ch, percent_fc, net_glob, is_print=True):
        self.before_epoch = int(
            self.first_train_ep * self.first_train_before_prune_ep_ratio
        )
        self.after_epoch = self.first_train_ep - self.before_epoch
        self.net.to(self.device)
        self.net.train()
        mask_ch_init = make_init_mask_ch(net_glob)
        mask_fc_init = make_init_mask_fc(net_glob)
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=self.momentum
        )
        epoch_loss = []
        mch1 = copy.deepcopy(self.mask_ch)
        mch2 = copy.deepcopy(self.mask_ch)

        mfc1 = copy.deepcopy(self.mask_fc)
        mfc2 = copy.deepcopy(self.mask_fc)
        print2log("---------------training before the prune--------------- ")
        for iteration in range(self.before_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.to(torch.int64)
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # Freezing Pruned weights by making their gradients Zero
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if iteration + 1 == self.before_epoch:
                cfg2, cfg_mask2, mch2, _ = fake_prune_ch(
                    percent_ch, copy.deepcopy(self.net), self.cfg_prune, self.ds
                )
                mfc2, _ = fake_prune_fc(
                    percent_fc,
                    copy.deepcopy(self.net),
                    copy.deepcopy(mask_ch_init),
                    copy.deepcopy(mask_fc_init),
                    self.ds,
                )  #
        if self.score_before_prune == True:
            self.score_fc = self.score_calculate_fc()
        #     self.score_ch = self.score_calculate_ch()

        if self.save_best:
            _, acc = self.eval_test(self.net)
            if acc > self.acc_best:
                self.acc_best = acc

        ## Un-Structured Prunning: Fully Connected Layers
        new_dict = real_prune_fc(
            copy.deepcopy(self.net),
            copy.deepcopy(self.mask_ch),
            copy.deepcopy(mfc2),
            self.ds,
        )  #
        self.net.load_state_dict(new_dict)
        if is_print:
            print(f"Un-Structured Pruned!")
        state_dict = new_dict
        final_mask_fc = mfc2

        self.net.load_state_dict(state_dict)

        ## Structured Prunning: Convolutional + BatchNorm Layers
        state_dict = copy.deepcopy(self.net.state_dict())
        final_mask_ch = copy.deepcopy(self.mask_ch)
        final_net = copy.deepcopy(self.net)
        new_net = real_prune_ch(
            copy.deepcopy(self.net),
            cfg2,
            cfg_mask2,
            self.ds,
            self.in_ch,
            self.device,
            self.args,
        )  #
        loss, acc = self.eval_test(new_net)
        # print2log(f"loss:{loss},acc:{acc}")
        if is_print:
            print(f"Structured Pruned!")
        # state_dict = new_dict
        state_dict = copy.deepcopy(new_net.state_dict())
        final_mc, final_mask_fc = update_mask_ch_fc(
            copy.deepcopy(self.mask_ch),
            copy.deepcopy(final_mask_fc),
            cfg_mask2,
            self.ds,
        )  #
        final_mask_ch = copy.deepcopy(final_mc)
        final_mask_fc = copy.deepcopy(final_mask_fc)
        final_net = copy.deepcopy(new_net)
        del new_net
        del self.net
        self.net = final_net
        self.mask_ch = final_mask_ch
        self.mch = mch2
        self.mask_fc = final_mask_fc
        # out= print_pruning(copy.deepcopy(self.net), net_glob, is_print)

        # self.pruned_total = out[0]
        # self.pruned_ch = out[1]
        # self.pruned_fc = out[2]
        # self.pruned_ch_rtonet = out[3]
        # self.pruned_fc_rtonet = out[4]
        print2log("---------------training after the prune--------------- ")
        self.net.train()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=self.momentum
        )
        for iteration in range(self.after_epoch):
            batch_loss = []
            # for name, param in net.named_parameters():
            # print(f'Name: {name}, NAN: {np.mean(np.isnan(param.detach().cpu().numpy()))}, INF: {np.mean(np.isinf(param.detach().cpu().numpy()))}')

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # Freezing Pruned weights by making their gradients Zero
                step_fc = 0
                for name, p in self.net.named_parameters():
                    if "weight" in name and "fc" in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        if step_fc == 0:
                            temp_mask = np.zeros_like(grad_tensor)
                            end_mask = self.mask_ch[-1].cpu().numpy()
                            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                            if idx0.size == 1:
                                idx0 = np.resize(idx0, (1,))

                            for i in range(len(idx0)):
                                ix0 = idx0[i]
                                size = self.ds * self.ds

                                temp_mask[:, i * size : i * size + size] = self.mask_fc[
                                    step_fc
                                ][:, ix0 * size : ix0 * size + size]
                            grad_tensor = grad_tensor * temp_mask
                        else:
                            grad_tensor = grad_tensor * self.mask_fc[step_fc]
                        p.grad.data = torch.from_numpy(grad_tensor).to(self.device)

                        step_fc = step_fc + 1

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.score_before_prune == False:
            self.score_fc = self.score_calculate_fc()
            # self.score_ch = self.score_calculate_ch()
        if self.save_best:
            _, acc = self.eval_test(self.net)
            if acc > self.acc_best:
                self.acc_best = acc
        print(len(self.mask_fc))

        return sum(epoch_loss) / len(epoch_loss)

    # def score_calculate_ch(model):
    #     score_ch = []
    #     for _, m in enumerate(model.modules()):
    #         if isinstance(m, nn.BatchNorm2d):
    #             weight_copy = m.weight.data.clone()
    #             tensor_list = weight_copy.reshape(-1).abs().tolist()
    #             min_val = min(tensor_list)
    #             max_val = max(tensor_list)
    #             normalized_list = np.array([(x - min_val)/(max_val - min_val) for x in tensor_list])
    #             score_ch.append(normalized_list)
    #     return score_ch

    def score_calculate_fc(self):
        score_fc = []
        para_list = []
        step = 0
        for _, m in enumerate(self.net.modules()):
            if isinstance(m, nn.Linear):
                weight_copy = m.weight.data.clone().cpu().reshape(-1).numpy()
                if step == 0 and self.ds != None:
                    k = 0
                    ks = self.ds
                    last_msk_ch = self.mask_ch[-1].cpu().numpy()
                    fc_mask = copy.deepcopy(self.mask_fc[step])
                    for j in range(len(last_msk_ch)):
                        if last_msk_ch[j] == 0:
                            fc_mask[:, j * ks * ks : j * ks * ks + ks * ks] = -1
                    flatten_mask = fc_mask.reshape(-1)
                    len_fc = len(flatten_mask)
                    score_list = np.ones(len_fc)
                    for i in range(len_fc):
                        if flatten_mask[i] == -1:
                            score_list[i] = 0
                        else:
                            score_list[i] = abs(weight_copy[k])
                            k += 1
                else:
                    flatten_mask = self.mask_fc[step].reshape(-1)
                    len_fc = len(flatten_mask)
                    score_list = np.zeros(len_fc)
                    for i in range(len_fc):
                        score_list[i] = abs(weight_copy[i])
                # score_list = weight_copy
                para_list.append(score_list)
                min_val = min(score_list)
                print(f"min_val:{min_val}")
                max_val = max(score_list)
                print(f"max_val:{max_val}")
                normalized_list = np.array(
                    [(x - min_val) / (max_val - min_val) for x in score_list]
                )
                # normalized_list_2d = normalized_list.reshape(m.weight.data.shape)
                score_fc.append(normalized_list)
                # print(f"len(normalized_list):{len(normalized_list)}")
                step += 1
        self.para_list = para_list
        return score_fc

    def score_calculate_ch(self):
        score_ch = []
        step = 0
        for _, m in enumerate(self.net.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                len_ch = len(self.mask_ch[step])
                score_list = np.ones(len_ch)
                k = 0

                for i in range(len_ch):
                    if self.mask_ch[step][i] == 0:
                        score_list[i] = 0
                    else:
                        score_list[i] = abs(weight_copy[k])
                        k += 1
                min_val = min(score_list)
                max_val = max(score_list)
                normalized_list = np.array(
                    [(x - min_val) / (max_val - min_val) for x in score_list]
                )
                score_ch.append(normalized_list)
                step += 1
        return score_ch

    def update_fc_score(self, ch_score, fc_score):
        glob_msk = copy.deepcopy(self.mask_fc[0])
        last_msk_ch = ch_score[-1]
        ks = self.ds
        for j in range(len(last_msk_ch)):
            if last_msk_ch[j] == 0:
                glob_msk[:, j * ks * ks : j * ks * ks + ks * ks] = -1
        glob_msk = glob_msk.flatten()
        for i in range(len(fc_score[0])):
            if glob_msk[i] == -1:
                fc_score[0][i] = 0
        return fc_score

    def update_fc_mask(self, ch_mask, fc_mask, score):
        glob_msk = copy.deepcopy(self.mask_fc[0])
        last_msk_ch = ch_mask[-1]
        ks = self.ds
        for j in range(len(last_msk_ch)):
            if last_msk_ch[j] == 0:
                glob_msk[:, j * ks * ks : j * ks * ks + ks * ks] = -1

        flatten_glob_msk = glob_msk.flatten()
        for i in range(len(flatten_glob_msk)):
            if flatten_glob_msk[i] == -1:
                fc_mask[0][i] = 0
                score[0][i] = 0
        return fc_mask, score

    def normal_train(self):
        print2log("---------------training before the prune--------------- ")
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=self.momentum
        )
        epoch_loss = []
        self.net.train()
        for iteration in range(self.local_ep):
            batch_loss = []
            # for name, param in net.named_parameters():
            # print(f'Name: {name}, NAN: {np.mean(np.isnan(param.detach().cpu().numpy()))}, INF: {np.mean(np.isinf(param.detach().cpu().numpy()))}')

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.to(torch.int64)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def train(self, percent_ch, percent_fc, net_glob, is_print=True):
        print2log("---------------training before the prune--------------- ")
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=self.momentum
        )
        epoch_loss = []
        self.net.train()
        for iteration in range(self.local_ep):
            batch_loss = []
            # for name, param in net.named_parameters():
            # print(f'Name: {name}, NAN: {np.mean(np.isnan(param.detach().cpu().numpy()))}, INF: {np.mean(np.isinf(param.detach().cpu().numpy()))}')

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.to(torch.int64)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        mch1 = copy.deepcopy(self.mask_ch)
        mch2 = copy.deepcopy(self.mask_ch)

        mfc1 = copy.deepcopy(self.mask_fc)
        mfc2 = copy.deepcopy(self.mask_fc)
        mask_ch_init = make_init_mask_ch(net_glob)
        mask_fc_init = make_init_mask_fc(net_glob)
        if self.score_before_prune == True:
            self.score_fc = self.score_calculate_fc()
        #     self.score_ch = self.score_calculate_ch()
        self.net.to(self.device)
        cfg2, cfg_mask2, mch2, _ = fake_prune_ch(
            percent_ch, copy.deepcopy(self.net), self.cfg_prune, self.ds
        )
        if is_print:
            print(f"New Model: {cfg2}")
        mfc2, _ = fake_prune_fc(
            percent_fc,
            copy.deepcopy(self.net),
            copy.deepcopy(mask_ch_init),
            copy.deepcopy(mask_fc_init),
            self.ds,
        )
        new_dict = real_prune_fc(
            copy.deepcopy(self.net),
            copy.deepcopy(mask_ch_init),
            copy.deepcopy(mfc2),
            self.ds,
        )
        self.net.load_state_dict(new_dict)
        if is_print:
            print(f"Un-Structured Pruned!")
        final_mask_fc = mfc2
        ## Structured Prunning: Convolutional + BatchNorm Layers
        final_mask_ch = copy.deepcopy(self.mask_ch)
        final_net = copy.deepcopy(self.net)
        new_net = real_prune_ch(
            copy.deepcopy(self.net),
            cfg2,
            cfg_mask2,
            self.ds,
            self.in_ch,
            self.device,
            self.args,
        )
        # loss, acc = self.eval_test(new_net)
        # print2log(f"loss:{loss},acc:{acc}")
        epoch_loss = []
        if is_print:
            print(f"Structured Pruned!")
        # state_dict = new_dict
        # print(f'self.mask: {self.mask}, cfg_mask2: {cfg_mask2}')
        final_mc, final_mfc = update_mask_ch_fc(
            copy.deepcopy((mask_ch_init)),
            copy.deepcopy(final_mask_fc),
            cfg_mask2,
            self.ds,
        )
        final_mask_ch = copy.deepcopy(final_mc)
        final_mask_fc = copy.deepcopy(final_mfc)
        final_net = copy.deepcopy(new_net)

        del new_net
        del self.net

        self.net = final_net
        self.mask_ch = final_mask_ch
        self.mask_fc = final_mask_fc
        self.mch = mch2
        # out= print_pruning(copy.deepcopy(self.net), net_glob, is_print)

        # self.pruned_total = out[0]
        # self.pruned_ch = out[1]
        # self.pruned_fc = out[2]
        # self.pruned_ch_rtonet = out[3]
        # self.pruned_fc_rtonet = out[4]
        self.net.train()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=self.momentum
        )
        for iteration in range(self.local_ep):
            batch_loss = []
            # for name, param in net.named_parameters():
            # print(f'Name: {name}, NAN: {np.mean(np.isnan(param.detach().cpu().numpy()))}, INF: {np.mean(np.isinf(param.detach().cpu().numpy()))}')

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                labels = labels.to(torch.int64)
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # Freezing Pruned weights by making their gradients Zero
                step_fc = 0
                for name, p in self.net.named_parameters():
                    if "weight" in name and "fc" in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        if step_fc == 0:
                            temp_mask = np.zeros_like(grad_tensor)
                            end_mask = self.mask_ch[-1].cpu().numpy()
                            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                            if idx0.size == 1:
                                idx0 = np.resize(idx0, (1,))

                            for i in range(len(idx0)):
                                ix0 = idx0[i]
                                size = self.ds * self.ds

                                temp_mask[:, i * size : i * size + size] = self.mask_fc[
                                    step_fc
                                ][:, ix0 * size : ix0 * size + size]
                            grad_tensor = grad_tensor * temp_mask
                        else:
                            grad_tensor = grad_tensor * self.mask_fc[step_fc]
                        p.grad.data = torch.from_numpy(grad_tensor).to(self.device)

                        step_fc = step_fc + 1

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.score_before_prune == True:
            self.score_fc = self.score_calculate_fc()
        #     self.score_ch = self.score_calculate_ch()
        if self.save_best:
            _, acc = self.eval_test(self.net)
            if acc > self.acc_best:
                self.acc_best = acc

        return sum(epoch_loss) / len(epoch_loss)

    def get_mask_ch(self):
        return self.mask_ch

    def get_mask_fc(self):
        return self.mask_fc

    def get_pruned_total(self):
        return self.pruned_total

    def get_pruned_ch(self):
        return self.pruned_ch

    def get_pruned_fc(self):
        return self.pruned_fc

    def get_pruned_ch_rtonet(self):
        return self.pruned_ch_rtonet

    def get_pruned_fc_rtonet(self):
        return self.pruned_fc_rtonet

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def get_state_dict(self):
        return self.net.state_dict()

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def get_best_acc(self):
        return self.acc_best

    def Sub_FedAVG_S(self, w_server, w_clients, masks_ch, masks_fc, model, ks, in_ch):
        """
        This function performs Sub-FedAvg-S (structured and unstructured pruning--Hybrid) as stated in the paper.
        This function updates the server model based on Sub-FedAvg. It is called at the end of each round.

        :param w_server: server model's state_dict
        :param w_clients: list of clients' model state_dict to be averaged
        :param masks_ch: list of clients' pruning masks of channels to be averaged
        :param masks_fc: list of clients' pruning masks of fcs to be averaged
        :param model: the original model model (net_glob)
        :param ks: kernel size of the model
        :param in_ch: number of input channel to the 1st layer

        :return w_server: updated server model's state_dict
        """
        ds = self.ds
        step_ch = 0
        step_fc = 0
        conv_layer = 1
        bn_layer = 1
        fc_layer = 1
        start_masks = [torch.ones(in_ch) for i in range(len(masks_ch))]
        end_masks = [masks_ch[i][step_ch] for i in range(len(masks_ch))]
        for m0 in model.modules():

            if isinstance(m0, nn.BatchNorm2d):
                # print(m0)
                name_weight = "main.bn{}.weight".format(bn_layer)
                name_bias = "main.bn{}.bias".format(bn_layer)
                name_running_mean = "main.bn{}.running_mean".format(bn_layer)
                name_running_var = "main.bn{}.running_var".format(bn_layer)

                names = [name_weight, name_bias, name_running_mean, name_running_var]

                for name in names:
                    # print(name)
                    weight_dev = w_server[name].device

                    count = np.zeros_like(w_server[name].data.cpu().numpy())
                    avg = np.zeros_like(w_server[name].data.cpu().numpy())

                    for i in range(len(masks_ch)):
                        count += end_masks[i].cpu().numpy()

                        idx0 = np.squeeze(
                            np.argwhere(np.asarray(end_masks[i].cpu().numpy()))
                        )
                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))

                        assert idx0.shape == w_clients[i][name].data.cpu().numpy().shape
                        assert (
                            avg[idx0.tolist()].shape
                            == w_clients[i][name].data.cpu().numpy().shape
                        )

                        for j in range(len(idx0)):
                            ix0 = idx0[j]
                            avg[ix0] += w_clients[i][name][j].data.cpu().numpy()

                    avg_reshape = avg.reshape([-1])
                    count_reshape = count.reshape([-1])
                    final_avg = np.divide(avg_reshape, count_reshape)

                    ind = np.isfinite(final_avg)

                    server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
                    server_reshape[ind] = final_avg[ind]

                    shape = w_server[name].data.cpu().numpy().shape
                    w_server[name].data = torch.from_numpy(
                        server_reshape.reshape(shape)
                    ).to(weight_dev)

                    # print(f'Name: {name}, NAN: {np.mean(np.isnan(server_reshape))}, INF: {np.mean(np.isinf(server_reshape))}')

                ## Updating step
                step_ch += 1

                start_masks = end_masks
                if step_ch < len(masks_ch[0]):
                    end_masks = [masks_ch[i][step_ch] for i in range(len(masks_ch))]
                ## Updating Layer
                bn_layer += 1

            elif isinstance(m0, nn.Conv2d):
                # print(m0)
                name_weight = "main.conv{}.weight".format(conv_layer)
                name_bias = "main.conv{}.bias".format(conv_layer)

                names = [name_weight, name_bias]

                for name in names:
                    # print(name)
                    weight_dev = w_server[name].device

                    avg = np.zeros_like(
                        w_server[name].data.cpu().numpy()
                    )  # [6, 3, 5, 5] [6]
                    count = np.zeros_like(
                        w_server[name].data.cpu().numpy()
                    )  # [6, 3, 5, 5] [6]

                    mm = np.zeros_like(
                        w_server[name].data.cpu().numpy()
                    )  # [6, 3, 5, 5] [6]
                    temp_masks = [np.zeros_like(mm) for _ in range(len(masks_ch))]

                    for i in range(len(masks_ch)):
                        temp_client = np.zeros_like(
                            w_server[name].data.cpu().numpy()
                        )  # [6, 3, 5, 5] [6]

                        idx0 = np.squeeze(
                            np.argwhere(np.asarray(start_masks[i].cpu().numpy()))
                        )
                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))

                        idx1 = np.squeeze(
                            np.argwhere(np.asarray(end_masks[i].cpu().numpy()))
                        )  # IND OUT
                        if idx1.size == 1:
                            idx1 = np.resize(idx1, (1,))

                        if name == name_weight:
                            print(
                                f"Client Shape: {w_clients[i][name].data.cpu().numpy().shape}, Supposed Shape: {[len(idx1),len(idx0), ks, ks]}"
                            )
                            assert w_clients[i][name].data.cpu().numpy().shape == (
                                len(idx1),
                                len(idx0),
                                ks,
                                ks,
                            )

                            for j in range(len(idx0)):  # [8, 3, 5, 5]
                                ix0 = idx0[j]
                                for k in range(len(idx1)):
                                    ix1 = idx1[k]
                                    # print(f'Server Shape: {avg[idx0, ix].shape} ')
                                    assert temp_client[ix1, ix0].shape == (ks, ks)
                                    assert w_clients[i][name][
                                        k, j
                                    ].cpu().numpy().shape == (ks, ks)

                                    temp_client[ix1, ix0] = (
                                        w_clients[i][name][k, j].cpu().numpy()
                                    )  # [out_channel, in_channel, k, k]
                                    temp_masks[i][ix1, ix0] = 1

                            non_zero_ind = np.nonzero(temp_masks[i].reshape([-1]))[0]
                            assert len(non_zero_ind) == len(
                                w_clients[i][name].data.cpu().numpy().reshape([-1])
                            )
                            # temp_masks[i][non_zero_ind.tolist()] = 1
                            count += temp_masks[i]
                            avg += temp_client

                        elif name == name_bias:
                            # print(f'Client Shape: {w_clients[i][name].data.cpu().numpy().shape}, Supposed Shape: {[len(idx1)]}')

                            assert w_clients[i][name].data.cpu().numpy().shape == (
                                len(idx1),
                            )

                            for j in range(len(idx1)):
                                ix1 = idx1[j]
                                temp_client[ix1] = (
                                    w_clients[i][name][j].data.cpu().numpy()
                                )
                                temp_masks[i][ix1] = 1

                            non_zero_ind = np.nonzero(temp_masks[i].reshape([-1]))[0]
                            assert len(non_zero_ind) == len(
                                w_clients[i][name].data.cpu().numpy().reshape([-1])
                            )
                            # temp_masks[i][non_zero_ind.tolist()] = 1
                            count += temp_masks[i]
                            avg += temp_client

                    avg_reshape = avg.reshape([-1])
                    count_reshape = count.reshape([-1])
                    # print(f'Name: {name}, AVG: {avg_reshape}, \n count_reshape: {count_reshape}')

                    final_avg = np.divide(avg_reshape, count_reshape)  # 6*3*5*5
                    # print(f'Name: {name}, Final AVG: {final_avg}')

                    ind = np.isfinite(final_avg)

                    server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
                    server_reshape[ind] = final_avg[ind]

                    shape = w_server[name].data.cpu().numpy().shape
                    w_server[name].data = torch.from_numpy(
                        server_reshape.reshape(shape)
                    ).to(weight_dev)

                    # print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF:
                    # {np.mean(np.isinf(w_server[name].cpu().numpy()))}')

                ## Updating Layer
                conv_layer += 1

            elif isinstance(m0, nn.Linear):
                # print(m0)

                name_weight = "fc{}.weight".format(fc_layer)
                name_bias = "fc{}.bias".format(fc_layer)

                names = [name_weight, name_bias]

                ## Weight
                name = names[0]
                weight_dev = w_server[name].device

                avg = np.zeros_like(w_server[name].data.cpu().numpy())  # [120, 400]
                count = np.zeros_like(w_server[name].data.cpu().numpy())

                for i in range(len(masks_fc)):
                    count += masks_fc[i][step_fc]

                    if fc_layer == 1:
                        temp_client = np.zeros_like(w_server[name].data.cpu().numpy())

                        idx0 = np.squeeze(np.argwhere(np.asarray(end_masks[i])))
                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))
                        print(w_clients[i][name].data.cpu().numpy().shape)
                        print(len(idx0))
                        # assert w_clients[i][name].data.cpu().numpy().shape == (50, len(idx0) * ds * ds)

                        for j in range(len(idx0)):
                            ix0 = idx0[j]
                            for k in range(ds * ds):
                                temp_client[:, ix0 * ds * ds + k] = (
                                    w_clients[i][name][:, j * ds * ds + k]
                                    .data.cpu()
                                    .numpy()
                                )

                        avg += temp_client

                    else:
                        avg += w_clients[i][name].data.cpu().numpy()

                avg_reshape = avg.reshape([-1])
                count_reshape = count.reshape([-1])

                final_avg = np.divide(avg_reshape, count_reshape)

                ind = np.isfinite(final_avg)

                server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
                server_reshape[ind] = final_avg[ind]

                shape = w_server[name].data.cpu().numpy().shape
                w_server[name].data = torch.from_numpy(
                    server_reshape.reshape(shape)
                ).to(weight_dev)

                # print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF:
                # {np.mean(np.isinf(w_server[name].cpu().numpy()))}')

                ## Bias
                name = names[1]
                avg = np.zeros_like(w_server[name].data.cpu().numpy())
                for i in range(len(masks_fc)):
                    avg += w_clients[i][name].data.cpu().numpy()

                final_avg = np.divide(avg, len(masks_fc))

                w_server[name].data = torch.from_numpy(final_avg).to(weight_dev)

                # print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF:
                # {np.mean(np.isinf(w_server[name].cpu().numpy()))}')

                ## Updating Layer
                fc_layer += 1
                step_fc += 1

        mask_ch_init = make_init_mask_ch(model)
        mask_fc_init = make_init_mask_fc(model)
        self.mask_ch = mask_ch_init
        self.mask_fc = mask_fc_init
        return w_server

    def Sub_FedAVG_U(self, w_server, w_clients, masks):
        """
        This function performs Sub-FedAvg-U (for unstructured pruning) as stated in the paper.
        This function updates the server model based on Sub-FedAvg. It is called at the end of each round.

        :param w_server: server model's state_dict
        :param w_clients: list of clients' model state_dict to be averaged
        :param masks: list of clients' pruning masks to be averaged

        :return w_server: updated server model's state_dict
        """
        step = 0
        for name in w_server.keys():

            if "weight" in name:

                weight_dev = w_server[name].device

                indices = []
                count = np.zeros_like(masks[0][step].reshape([-1]))
                avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
                for i in range(len(masks)):
                    count += masks[i][step].reshape([-1])
                    avg += w_clients[i][name].data.cpu().numpy().reshape([-1])

                final_avg = np.divide(avg, count)
                ind = np.isfinite(final_avg)

                temp_server = w_server[name].data.cpu().numpy().reshape([-1])
                temp_server[ind] = final_avg[ind]

                # print(f'Name: {name}, NAN: {np.mean(np.isnan(temp_server))}, INF: {np.mean(np.isinf(temp_server))}')

                shape = w_server[name].data.cpu().numpy().shape
                w_server[name].data = torch.from_numpy(temp_server.reshape(shape)).to(
                    weight_dev
                )

                step += 1
            else:

                avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
                for i in range(len(masks)):
                    avg += w_clients[i][name].data.cpu().numpy().reshape([-1])
                avg /= len(masks)

                # print(f'Name: {name}, NAN: {np.mean(np.isnan(avg))}, INF: {np.mean(np.isinf(avg))}')
                weight_dev = w_server[name].device
                shape = w_server[name].data.cpu().numpy().shape
                w_server[name].data = torch.from_numpy(avg.reshape(shape)).to(
                    weight_dev
                )
        # mask_ch_init = make_init_mask_ch(model)
        # mask_fc_init = make_init_mask_fc(model)
        # self.mask_ch = mask_ch_init
        # self.mask_fc = mask_fc_init
        return w_server

    def eval_test(self, model):
        model.to(self.device)
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.ldr_test:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                # 累加损失
                test_loss += F.cross_entropy(
                    outputs, labels.long(), reduction="sum"
                ).item()

                # 获取预测结果
                _, predicted = torch.max(outputs, 1)

                # 统计预测正确的数量
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # 计算平均损失
        test_loss /= total

        # 计算准确率
        accuracy = 100.0 * correct / total

        return test_loss, accuracy

    def eval_train(self, model):
        model.to(self.device)
        model.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                train_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100.0 * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
