import os
import numpy as np
import sklearn.model_selection as ms
from sklearn.naive_bayes import GaussianNB
from users.models import MyUser
import datetime
import random
import operator
from pyecharts import Bar
from pyecharts import Line, Pie
from featureselection.models import TaskResult, Task, Recall, F1score, Statistics
from sklearn import svm
from sklearn.metrics import hinge_loss, f1_score, recall_score, precision_score
# write the sourcefile to the destination directory


# 将源文件写入目标文件
def write2file(sourcefile, destfileaddress):
    destination = open(os.path.join(destfileaddress, sourcefile.name), 'wb+')
    for chunk in sourcefile.chunks():
        destination.write(chunk)
    destination.close()
    return 'successfully uploaded'


# according to file_address load the csv file, return the data and the label
def load_csv(file_address):
    out = open(file_address, 'rb')
    try:
        file = np.loadtxt(out, dtype=str, delimiter=',', skiprows=0)
        out.close()
        a, b = file.shape
        dict = {}  # 对应的每一列的非数字型值的对应编号
        lis = []  # 用作构造新的数字矩阵
        for i in range(b):
            dic = {}
            count = 0
            l = file[:, i]
            newl = l.tolist()
            c = True
            not_used = [i for i in range(200)]
            cc = True
            for j in range(a):
                try:
                    int(file[j, i])
                except BaseException:
                    cc = False
            for j in range(a):
                if cc:
                    newl[j] = int(file[j, i])
                else:
                    try:
                        f = float(file[j, i])
                        if c:
                            try:
                                # 说明已经是数字了
                                intt = int(file[j, i])
                                dic[file[j, i]] = intt
                                not_used.remove(intt)
                            except BaseException:
                                if dic:  # 如果里面有内容了，则往里填充就行
                                    dic[file[j, i]] = f
                                else:
                                    newl[j] = f  # 是一个浮点数，此列不用管，因为非数字往往与整数一起出现
                                    c = False
                        else:
                            newl[j] = f
                    except BaseException:
                        if file[j, i] not in dic:
                            dic[file[j, i]] = -1  # 原赋值为count
                            count += 1
                    # newl[j] = dic[file[j, i]]
            if dic:
                for k, v in dic.items():
                    if dic[k] == -1:  # 说明k的值为-1
                        dic[k] = not_used[0]
                        not_used.pop(0)
                for j in range(a):
                    newl[j] = dic[newl[j]]
            dict[i] = dic
            lis.append(newl)
        newfile = np.array(lis)
        newfile = newfile.T
        data = newfile[:, 0:-1]
        labels = newfile[:, -1]
        return data, labels, dict
    except BaseException:
        out.close()
        raise BaseException


# 如其名：一个简单的载入csv文件的函数
def simple_load_csv(file_address):
    out = open(file_address, 'rb')
    file = np.loadtxt(out, dtype=str, delimiter=',', skiprows=0)
    out.close()
    data = file[:, 0:-1]
    labels = file[:, -1]
    return data, labels


# ————功能：对结果进行准确度分析，参数为：实际的标签值 和 预测的标签值
def accuracy(test_lables, pre_lables):
    correct = np.sum(test_lables == pre_lables)
    test_num = len(test_lables)
    return float(correct) / test_num


# ————贝叶斯分类器，训练集：验证集 = 5:1，返回结果为准确率
def naviebayes(data, labels):
    kf = ms.KFold(5, True, None)
    clf = GaussianNB()
    result_set = []
    sum_f1 = 0
    sum_recall = 0
    sum_accuracy = 0
    count = 0
    # result_set = [(clf.fit(data[train], labels[train]).predict(data[test]), labels[test]) for train, test in kf.split(data)]
    for train, test in kf.split(data):
        predict = clf.fit(data[train], labels[train]).predict(data[test])
        sum_accuracy += precision_score(labels[test], predict, average='weighted')
        sum_f1 += f1_score(labels[test], predict, average='weighted')
        sum_recall += recall_score(labels[test], predict, average='weighted')
        # result_set.append((predict, labels[test]))
        count += 1
    # scores = [accuracy(result[1], result[0]) for result in result_set]
    # for score in scores:
    #    sum_accuracy += float(score)
    return sum_accuracy / count, sum_recall / count, sum_f1 / count


# ————功能：把数据集按照特征选择结果进行切分
def change_datashape(data, fs_result):
    mylist = []
    listfs_result = fs_result.tolist()
    listfs_result.sort()
    for i in listfs_result:
        i = int(i)
        mylist.append(data[:, i].tolist())
    newdata = (np.array(mylist)).T
    return newdata


# CHI结果分析，并将结果写入文件中
def chians_analyze(chi_ans, data, labels, username):
    accu = []  # 精确度
    rec = []
    f_1 = []
    for i in chi_ans:
        narray = np.array(i)
        newdata = change_datashape(data, narray)
        a, recall, f1 = naviebayes(newdata, labels)
        a = round(a * 100, 2)
        recall = round(recall * 100, 2)
        f1 = round(f1 * 100, 2)
        a = str(a)
        accu.append(a)
        rec.append(recall)
        f_1.append(f1)
    user = MyUser.objects.get(username=username)
    chians_add = user.chians_address
    num = random.randint(1, 255)  # 进一步减小文件名重叠
    filename = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + str(num) + '.txt'
    file = open(chians_add + filename, 'w')
    for i in range(len(chi_ans)):
        if i != 0:
            file.write('\n')
        file.write(list2str(chi_ans[i])+','+accu[i]+',NotExport,'+str(rec[i])+','+str(f_1[i]))
    file.close()
    return filename


# CHI算法真实的结果，根据每个特征的CHI值计算所得
def chi_real_result(chi):
    chilist = chi.tolist()
    dic = {}
    for i in range(len(chilist)):
        dic[i] = chilist[i]
    sorted_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    ans = []
    for i in range(len(sorted_dic)):
        ans.append(sorted_dic[i][0])
    newans = np.array(ans)
    return newans


def list2str(lis):  # 这个str数据与数据之间为空格
    strlis = ''
    for i in range(len(lis)):
        s = str(lis[i])
        if i != 0:
            strlis += ' '
        strlis += s
    return strlis


def str2list(strlis):  # 字符串转化为链表
    lis = []
    strlis = strlis.strip()
    s = ''
    for i in strlis:
        if i == ' ' or i == ',':
            s = s.strip()
            if s != '':
                lis.append(s)
            s = ''
        else:
            s += i
    s = s.strip()
    if s != '':
        lis.append(s)
    return lis


#  字典转化为字符串
def dic2str(dict):
    s = ''
    i = 0
    for key, value in dict.items():
        if i != 0:
            s += ','
            s += key
        else:
            s += key
            i = 1
    return s


#  得到数据集的列数, 参数num表示第几次运行这个函数
def get_column_num(file, num):
    if num == 1:  # 第一次运行函数将更新temp.csv
        with open('C:\\Users\\HZF\\Desktop\\fsSelectionProject\\temp.csv', 'wb+') as destination:
            line = file.readline()
            destination.write(line)
            destination.close()
    out = open('C:\\Users\\HZF\\Desktop\\fsSelectionProject\\temp.csv', 'rb')
    tempfile = np.loadtxt(out, dtype=str, delimiter=',', skiprows=0)
    out.close()
    b = tempfile.shape
    try:
        b = b[0]
    except BaseException:  # 此时b为()
        b = 1
    return b


#  用户自选特征选择的柱状图
def echarts_bar(userresult_list):
    _data = []
    recall_data = []
    f1_data = []
    _x = []
    bar = Bar('User-selected Evaluation', width=750, height=500)
    for i in userresult_list:
        _x.append(i.user_result)
        _data.append(i.accuracy)
        recall_data.append(i.recall)
        f1_data.append(i.f1)
    bar.add('Accuracy（%)', _x, _data, yaxis_min=0, yaxis_max=100, area_color='#4575b4')
    bar.add('Recall（%)', _x, recall_data, yaxis_min=0, yaxis_max=100, area_color='#74add1')
    bar.add('F-1（%)', _x, f1_data, yaxis_min=0, yaxis_max=100, area_color='#74add1')
    return bar


#  alpha，CFS算法的柱状图
def algorithm_bar(result, task_id):
    _data2 = []
    _data3 = []
    _data4 = []
    _data5 = []
    recall = Recall.objects.get(task_id=task_id)
    f1 = F1score.objects.get(task_id=task_id)
    _x = ['Original Subset', 'CFS:   '+result.cfs_result, 'alpha_investing:   '+result.alphainvesting_result]
    bar = Bar('Evaluation：')
    _data2.append(result.original_accuracy)
    _data2.append(result.cfs_accuracy)
    _data2.append(result.alphainvesting_accuracy)
    _data3.append(0)
    _data3.append(result.cfs_time)
    _data3.append(result.alphainvesting_time)
    _data4.append(recall.original)
    _data4.append(recall.cfs)
    _data4.append(recall.alpha)
    _data5.append(f1.original)
    _data5.append(f1.cfs)
    _data5.append(f1.alpha)
    bar.add('Accuracy（%)', _x, _data2, area_color='#4575b4')
    bar.add('Runtime（ms)', _x, _data3, area_color='#74add1')
    bar.add('Recall（%)', _x, _data4, area_color='#74add1')
    bar.add('F-1（%)', _x, _data5, area_color='#74add1')
    return bar


#  CHI化成折线图
def chi_line(task_id):
    task_reuslt = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    user = MyUser.objects.get(username=task.user)
    chi_file_address = user.chians_address
    out = open(chi_file_address + task_reuslt.chi_file, 'rb')
    try:
        file = np.loadtxt(out, delimiter=',', dtype=str, skiprows=0)
        out.close()
        data1 = file[:, 0].tolist()
        data2 = file[:, 1].tolist()
        data3 = file[:, 3].tolist()
        data4 = file[:, 4].tolist()
        line = Line('Evaluation Of Chi2：')
        line.add('Accuracy（%)', data1, data2, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
        line.add('Recall（%)', data1, data3, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
        line.add('F-1（%)', data1, data4, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
        return line, file
    except BaseException:
        print('由于数据集中有负数值，无CHI算法结果')
        return None, None


def chi_result(task_id, index2, trigger, export):  # trigger是判断是否是需要重写回目标文件，功能不同
    if trigger:
        task_reuslt = TaskResult.objects.get(task_id=task_id)
        task = Task.objects.get(id=task_id)
        user = MyUser.objects.get(username=task.user)
        chi_file_address = user.chians_address
        out = open(chi_file_address+task_reuslt.chi_file, 'rb')
        file = np.loadtxt(out, delimiter=',', dtype=str, skiprows=0)
        out.close()
        file = str2list(file[index2, 0])
        return file
    else:
        task_reuslt = TaskResult.objects.get(task_id=task_id)
        task = Task.objects.get(id=task_id)
        user = MyUser.objects.get(username=task.user)
        chi_file_address = user.chians_address
        out = open(chi_file_address + task_reuslt.chi_file, 'rb')
        file = np.loadtxt(out, delimiter=',', dtype=str, skiprows=0)
        out.close()
        file[index2, 2] = export
        np.savetxt(chi_file_address + task_reuslt.chi_file, file, fmt='%s', delimiter=',')
        return file


# np的数组转化为字典
def np2dic(nparray):
    dic = {}
    a = nparray.shape
    for i in range(a[0]):
        dic[i] = nparray[i, :].tolist()
    return dic


def rlf_line(rlf_set):
    x = []
    y = []
    yr = []
    yf = []
    for i in rlf_set:
        x.append(i.rlf_result)
        y.append(i.rlf_accuracy)
        yr.append(i.rlf_recall)
        yf.append(i.rlf_f1)
    line = Line('Evaluation Of ReliefF：')
    line.add('Accuracy（%)', x, y, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
    line.add('Recall（%)', x, yr, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
    line.add('F-1（%)', x, yf, yaxis_min=0, yaxis_max=100, mark_point=["max", "min"],)
    return line


def feature_set_length(feature_set):
    count = 0
    for i in feature_set:
        if i == ' ':
            count += 1
    count += 1
    return count


# 系统的最佳特征子集选择函数，参数是字典，格式为{ algorithm_name:{feature_set:[accuracy, recall, f1, (opinion_score)]} },输出最佳特征子集：feature_set
# opinion_score = k * 1 / length(feature_set) + accuracy + recall + f1   (k的值通过计算得到约为 0.625)
def select_the_bestset(dictionary):
    dic = {}
    for k, v in dictionary.items():
        for k1, v1 in v.items():
            opinion_score = 0.625 * 1 / feature_set_length(k1) + v1[0] + v1[1] + v1[2]
            v1.append(opinion_score)
        newdictionary = sorted(v.items(), key=lambda value: value[1][3], reverse=True)
        dic[k] = newdictionary[0]
    newd = sorted(dic.items(), key=lambda item: item[1][1][3], reverse=True)
    return newd[0]


def statistics_pie_chart():
    algorithm_list = ['CFS', 'alpha-investing', 'Chi2', 'ReliefF', "User's"]
    try:
        ans = Statistics.objects.get(id=1)
        num_list = []
        num_list.append(ans.CFS_num)
        num_list.append(ans.Alpha_num)
        num_list.append(ans.Chi2_num)
        num_list.append(ans.ReliefF_num)
        num_list.append(ans.User_num)
        sum1 = ans.CFS_num + ans.Alpha_num + ans.Chi2_num + ans.ReliefF_num + ans.User_num
        pie = Pie('')
        pie.add('', algorithm_list, num_list, center=[50, 50], radius=[20, 75], rosetype='radius')
    except BaseException:
        Statistics.objects.create(id=1)
        sum1 = 0
        pie = None
    return sum1, pie
