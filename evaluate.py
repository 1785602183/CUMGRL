import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models import LogReg
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
np.random.seed(0)
def evaluate(embeds, idx_train, idx_val, idx_test, labels, device, isTest=True, filename='result.txt'):
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []  ##

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        val_accs = [];
        test_accs = []
        val_micro_f1s = [];
        test_micro_f1s = []
        val_macro_f1s = [];
        test_macro_f1s = []
        for iter_ in range(50):
            # train
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            #           print(val_lbls)

            # print(preds)
            #  print("轮次{}, epoch{}".format(_,iter_))
            # print('验证集预测标签种类',len(set(preds.cpu().numpy().tolist())))
            # print('验证集真实标签种类',len(set(val_lbls.cpu().numpy().tolist())))

            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            #     print('测试集预测标签种类',len(set(preds.cpu().numpy().tolist())))
            #    print('测试集真实标签种类',len(set(test_lbls.cpu().numpy().tolist())))
            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
        # print('测试集的acc和f1',test_acc.item(),test_f1_macro,test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])  ###

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
    print(filename)
    file_classify = open(filename + 'classify.txt', 'a+')

    file_sim = open(filename + 'sim.txt', 'a+')
    if isTest:
        # print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
        #                                                                                         np.std(macro_f1s),
        #                                                                                         np.mean(micro_f1s),
        #                                                                                         np.std(micro_f1s)),file=file_classify)
        print("\t {:.4f} ({:.4f})  {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                           np.std(macro_f1s),
                                                           np.mean(micro_f1s),
                                                           np.std(micro_f1s)), file=file_classify
              )

    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())

    run_kmeans(test_embs, test_lbls, nb_classes, test_lbls, filename)
    run_similarity_search(test_embs, test_lbls, file_sim)
    file_classify.close()

    file_sim.close()


def run_similarity_search(test_embs, test_lbls, file_sim):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows, N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    st = ','.join(st)
    # print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))

    print("\t{}".format(st), file=file_sim)


def precision(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * TP / (TP + FP)


def recall(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * TP / (TP + FN)


def F_measure(result, label, beta=1):
    prec = precision(result, label)
    r = recall(result, label)
    return (beta * beta + 1) * prec * r / (beta * beta * prec + r)


def contingency_table(result, label):
    total_num = len(label)

    TP = TN = FP = FN = 0
    for i in range(total_num):
        for j in range(i + 1, total_num):
            if label[i] == label[j] and result[i] == result[j]:
                TP += 1
            elif label[i] != label[j] and result[i] != result[j]:
                TN += 1
            elif label[i] != label[j] and result[i] == result[j]:
                FP += 1
            elif label[i] == label[j] and result[i] != result[j]:
                FN += 1
    return (TP, TN, FP, FN)


def rand_index(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * (TP + TN) / (TP + FP + FN + TN)


from scipy.optimize import linear_sum_assignment


def clusteringAcc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(list(zip(*ind)))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def run_kmeans(x, y, k, labels, filename):
    estimator = KMeans(n_clusters=k)
    # labels =torch.argmax(labels[0, :], dim=1).cpu().numpy()
    file_nmi = open(filename + 'nmi.txt', 'a+')
    file_acc = open(filename + 'acc.txt', 'a+')
    # file_ari = open(filename + 'ari.txt', 'a+')
    # file_ri = open(filename + 'ri.txt', 'a+')
    # file_ami = open(filename + 'ami.txt', 'a+')
    # file_f1 = open(filename + 'F1.txt', 'a+')
    NMI_list = []
    acc_list = []
    # ARI_list = []
    # RI_list = []
    # AMI_list = []
    # F1_list = []
    from sklearn import metrics
    for i in range(5):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s1 = normalized_mutual_info_score(y, y_pred)
        NMI_list.append(s1)
        acc = clusteringAcc(y, y_pred)
        acc_list.append(acc)
    #   print(acc,s1)
    # s1 = metrics.adjusted_mutual_info_score(labels, y_pred)
    # AMI_list.append(s1)
    # s1 = metrics.adjusted_rand_score(labels, y_pred)
    # ARI_list.append(s1)
    # s1 = rand_index(y_pred,labels)
    # RI_list.append(s1)
    #
    # s1 = F_measure(y_pred,labels)
    # F1_list.append(s1)

    # s1 = sum(NMI_list) / len(NMI_list)
    # # print('\t[Clustering] NMI: {:.4f}'.format(s1))
    # print('\t{:.4f}'.format(s1), file=file_nmi)
    # s1 = sum(AMI_list) / len(AMI_list)
    # # print('\t[Clustering] NMI: {:.4f}'.format(s1))
    # print('\t{:.4f}'.format(s1), file=file_ami)
    # s1 = sum(ARI_list) / len(ARI_list)
    # # print('\t[Clustering] NMI: {:.4f}'.format(s1))
    # print('\t{:.4f}'.format(s1), file=file_ari)
    s1 = sum(NMI_list) / len(NMI_list)
    print('{:.4f}'.format(s1), file=file_nmi)
    s1 = sum(acc_list) / len(acc_list)
    print('{:.4f}'.format(s1), file=file_acc)
    # print('\t{:.4f}'.format(s1), file=file_ri)
    #
    # s1 = sum(F1_list) / len(F1_list)
    # # print('\t[Clustering] NMI: {:.4f}'.format(s1))
    # print('\t{:.4f}'.format(s1), file=file_f1)

    file_nmi.close()
    file_acc.close()
    # file_ari.close()
    # file_ri.close()
    # file_ami.close()
    # file_f1.close()


import collections


def purity(result, label):
    # 计算纯度
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)
    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:  # 求交集
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)

    return sum(t) / total_num