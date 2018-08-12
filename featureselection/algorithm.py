from skfeature.function.streaming.alpha_investing import alpha_investing
from skfeature.function.statistical_based import CFS as CFS
from skfeature.function.statistical_based.chi_square import chi_square
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.similarity_based.reliefF import feature_ranking
import numpy as np
import time


# ————function：alp
def alpinvesting(data, labels):
    start = time.clock()
    ans = alpha_investing(data, labels, w0=5, dw=1)
    end = time.clock()
    time_using = (end - start) * 1000
    time_using = round(time_using, 3)
    return ans, time_using


# ———--function：Correlation-based Feature Selection;
# But when feature_number is small ,the algorithm will generate the -1
def cfs(data, labels):
    start = time.clock()
    ans = CFS.cfs(data, labels)
    end = time.clock()
    lans = ans.tolist()  # 当原数据集的特征数低于五个时，cfs算法将产生额外的‘-1’特征指数，删除即可。
    index = len(lans)
    for a in range(len(lans)):
        if lans[a] == -1:
            index = a
            break
    lans = lans[:index]
    time_using = (end - start) * 1000
    time_using = round(time_using, 3)
    return np.array(lans), time_using


# ————function：according to CHI, and then sort the weight of the each feature
def CHI_square(data, labels):
    start = time.clock()
    ans = chi_square(data, labels)  # 与下面的函数同一种功能type is ndarray
    # print(chis.feature_ranking(chinum))# 这个函数是sklearn中的函数
    end = time.clock()
    time_using = (end - start) * 1000
    time_using = round(time_using, 3)
    return ans, time_using


# -------function: according to reliefF
def relief_f(data, labels):
    start = time.clock()
    ans = reliefF(data, labels)
    ans = feature_ranking(ans)
    end = time.clock()
    time_using = (end - start) * 1000
    time_using = round(time_using, 3)
    return ans, time_using
