import numpy as np

#混淆矩阵
class Confusion_Matrix:
    def __init__(self, real, pre):
        self.real = real
        self.pre = pre
    def con_mat(self):
        idx_real = np.argmax(self.real, axis=1)
        idx_pre = np.argmax(self.pre, axis=1)
        con_matrix = np.zeros((10, 10))
        for i in range(len(idx_real)):
            x = idx_pre[i]
            y = idx_real[i]
            con_matrix[x][y] += 1
        return con_matrix

class Performance_Index:
    def __init__(self, con_mat):
        self.con_mat = con_mat

    def per_index(self):
        FP = self.con_mat.sum(axis=0) - np.diag(self.con_mat)
        FN = self.con_mat.sum(axis=1) - np.diag(self.con_mat)
        TP = np.diag(self.con_mat)
        TN = self.con_mat.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        return FP, FN, TP, TN
    def Acc_all(self):
        acc_all = sum(np.diag(self.con_mat)) / self.con_mat.sum()
        return acc_all
    def Acc(self):
        FP, FN, TP, TN = self.per_index()
        acc = TP / (TP + FP)
        return acc
    def Precision(self):
        FP, FN, TP, TN = self.per_index()
        precision = TP / (TP + FP)
        return precision
    def Re_call(self):
        FP, FN, TP, TN = self.per_index()
        re_call = TP / (TP + FN)
        return re_call