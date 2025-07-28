import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from torch import tensor, optim, nn
from miscellaneous import print2log
import torch.nn.functional as F


def train_net(net, dataloader_train, dataloader_test, device='cpu', lr=0.1, optimizer='sgd', epochs=5, rho=0, reg=1e-5):
    # 计算训练前的模型准确率
    net.to(device)
    train_acc = compute_accuracy(net, dataloader_train, device=device)
    test_acc = compute_accuracy(net, dataloader_test, device=device)
    print2log(
        '>> Pre-Training Training accuracy: {0:.5f}, Pre-Training Test accuracy: {1:.5f}'.format(train_acc, test_acc))

    # 设置优化器
    if optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg)
    elif optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg, amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=rho, weight_decay=reg)
    criterion = nn.CrossEntropyLoss().to(device)

    # 训练
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(dataloader_train):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print2log('>> Epoch: {0} Loss: {1:.5f}'.format(epoch + 1, epoch_loss))

    train_acc = compute_accuracy(net, dataloader_train, device=device)
    test_acc = compute_accuracy(net, dataloader_test, device=device)

    print2log('>> Training Complete, Training accuracy: {0:.5f}, Test accuracy: {1:.5f}'.format(train_acc, test_acc))
    return train_acc, test_acc


def train_net_fedprox(net, global_net, train_dataloader, test_dataloader, mu=1, device='cpu', lr=0.1, optimizer='sgd',
                      epochs=5, rho=0, reg=1e-5):
    net.to(device)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print2log(
        '>> Pre-Training Training accuracy: {0:.5f}, Pre-Training Test accuracy: {1:.5f}'.format(train_acc, test_acc))

    if optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg)
    elif optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg,
                               amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=rho,
                              weight_decay=reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print2log('>> Epoch: {0} Loss: {1:.5f}'.format(epoch + 1, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    print2log('>> Training Complete, Training accuracy: {0:.5f}, Test accuracy: {1:.5f}'.format(train_acc, test_acc))

    return train_acc, test_acc


def eval_model(test_loader,local_model,device):
        local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        count = 0
        for batch_id, batch in enumerate(test_loader):
            data, target = batch
            dataset_size += data.size()[0]
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            output = local_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            if count < 0:
                print(pred)
                count += 1

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

def fuse_model_by_teachers(local_model,device,train_loader,test_loader,models):

    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.02, momentum=0.9)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.KLDivLoss(reduction="batchmean")
    local_model.to(device)
    for i in range(len(models)):
        models[i][1] = models[i][1].train()
        models[i][1] = models[i][1].to(device)

    count = 0
    epochs = 1
    for e in range(epochs):
        local_model.train()
        total_loss = 0.0

        for batch_id, batch in enumerate(train_loader):
            data, target = batch
            data = data.to(device)
            target = target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            student_output = local_model(data)
            T = 1
            alpha = 0.5
            beta = 0.5
            teacher_output = torch.zeros_like(student_output)
            weight = 1 / len(models)
            with torch.no_grad():
                for _, model in models:
                    teacher_output += weight * model(data)
            loss1 = criterion1(student_output, target)
            loss2 = criterion2(F.log_softmax(student_output / T, dim=1),
                                F.softmax(teacher_output / T, dim=1)) * T * T

            if e == 1 and count < 2:
                print2log("loss1:{}, alpha*loss1:{}; loss2:{}, beta*loss2:{}".format(loss1, alpha * loss1, loss2, beta*loss2))
                count += 1

            loss = alpha * loss1 + beta * loss2
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            optimizer.step()

        acc, total_l = eval_model(test_loader,local_model,device)
        print("Epoch {} done. Train loss {}. Valid accuracy {}".format(e, total_loss, acc))
    return acc, total_l

def compute_accuracy(net, dataloader, get_confusion_matrix=False, device="cpu"):
    net.to(device)
    was_training = False
    if net.training:
        net.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device, dtype=torch.int64)
            out = net(x)
            _, pred_label = torch.max(out.data, 1)

            # if batch_idx == 0:
            #     print('In learning....', target)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        net.train()

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return correct / float(total), conf_matrix

    return correct / float(total)