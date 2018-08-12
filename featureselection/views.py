import os

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .forms import TaskForm
from .models import Task
from django.http import HttpResponseRedirect
from django.urls import reverse
from .backend import write2file
from .models import MyUser
import featureselection.backend as backend
import featureselection.algorithm as algorithm
from featureselection.models import TaskResult, Recall, F1score, Statistics
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import StreamingHttpResponse
import numpy as np
from featureselection.models import UsersResult
from featureselection.models import ReliefFResult
from django.core.exceptions import ValidationError
from django import forms
# Create your views here.


def index(request):
    return render(request, 'featureselection/index.html')


@login_required
def add_task(request):
    if request.method != 'POST':
        form = TaskForm()
    else:
        form = TaskForm(request.POST, request.FILES)
        form.user = request.POST.get('username')
        if form.is_valid():
            feature_names = request.POST.get('feature_names')
            file = request.FILES.get('upload_file')
            feature_names_list = backend.str2list(feature_names)
            column_num = backend.get_column_num(file, 2)
            username = request.POST.get('username')
            user = MyUser.objects.get(username=username)
            #  自定义上传文件的默认地址
            #  default_storage.save(user.upload_address + file.name, ContentFile(file.read()))
            write2file(file, user.upload_address)  # write the file to upload directory
            try:
                data, labels, dict = backend.load_csv(user.upload_address + file.name)
                label_value = dict[column_num - 1]
                label_values = backend.dic2str(label_value)
                task_id = form.save(user, file.name, label_values).id
                return HttpResponseRedirect(reverse('featureselection:task_result', args=(task_id,)))
            except BaseException:
                form.add_error(None, ValidationError('There Is An Error In The File. Please Check And Try Again.'))
                if os.path.exists(user.upload_address + file.name):
                    os.remove(user.upload_address + file.name)
    context = {'form': form}
    return render(request, 'featureselection/task.html', context)


@login_required
def task_result(request, task_id):  # 修改...
    task = Task.objects.get(id=task_id)
    filename = str(task.upload_file)
    user = MyUser.objects.get(username=task.user)
    upload_add = user.upload_address
    a = upload_add + filename
    try:
        data, labels, dict = backend.load_csv(a)
        taskresult = TaskResult()
        if task.task_type == 'classification':
            cfs_ans, cfs_time = algorithm.cfs(data, labels)
            cfs_data = backend.change_datashape(data, cfs_ans)
            alp_ans, alp_time = algorithm.alpinvesting(data, labels)
            alp_data = backend.change_datashape(data, alp_ans)
            try:
                chi, chi_time = algorithm.CHI_square(data, labels)
                chi_real = backend.chi_real_result(chi)
                chi_ans = []  # 保存的是chi的每个结果
                temp = []
                for i in chi_real.tolist():
                    temp.append(i)
                    temp1 = []
                    for j in temp:
                        temp1.append(j)
                    chi_ans.append(temp1)
                chi_file = backend.chians_analyze(chi_ans, data, labels, user.username)  # 分析得到存储chi算法得出的结果的文件

                taskresult.chi_result = backend.list2str(chi_real.tolist())
                taskresult.chi_file = chi_file
                taskresult.chi_time = chi_time
            except ValueError:
                taskresult.chi_file = ''
                os.remove(user.chians_address + chi_file)
                print('数据中存在负数')

            rlf, rlf_time = algorithm.relief_f(data, labels)
            rlf_ans = []  # 保存rlf每个特征子集的结果
            temp = []
            for i in rlf:  # 保存rlf
                temp.append(i)
                temp1 = []
                for j in temp:
                    temp1.append(j)
                rlf_ans.append(temp1)
            for i in rlf_ans:
                mylist = []
                i.sort()
                for j in i:
                    j = int(j)
                    mylist.append(data[:, j].tolist())
                rlf_reuslt = backend.list2str(i)
                newdata = (np.array(mylist)).T
                rlf_accuracy, rlf_recall, rlf_f1 = backend.naviebayes(newdata, labels)
                rlf_accuracy = round(rlf_accuracy * 100, 2)
                rlf_recall = round(rlf_recall * 100, 2)
                rlf_f1 = round(rlf_f1 * 100, 2)
                relief_result = ReliefFResult()  # 为每个记录
                relief_result.task = task
                relief_result.rlf_result = rlf_reuslt
                relief_result.rlf_accuracy = rlf_accuracy
                relief_result.rlf_export = 'NotExport'
                relief_result.rlf_recall = rlf_recall
                relief_result.rlf_f1 = rlf_f1
                relief_result.save()

            original_accuracy, original_recall, original_f1 = backend.naviebayes(data, labels)
            original_accuracy = round(original_accuracy * 100, 2)
            original_recall = round(original_recall * 100, 2)
            original_f1 = round(original_f1 * 100, 2)
            cfs_accuracy, cfs_recall, cfs_f1 = backend.naviebayes(cfs_data, labels)
            cfs_accuracy = round(cfs_accuracy * 100, 2)
            cfs_recall = round(cfs_recall * 100, 2)
            cfs_f1 = round(cfs_f1 * 100, 2)
            alp_accuracy, alp_recall, alp_f1 = backend.naviebayes(alp_data, labels)
            alp_accuracy = round(alp_accuracy * 100, 2)
            alp_recall = round(alp_recall * 100, 2)
            alp_f1 = round(alp_f1 * 100, 2)

            recall = Recall()  # 保存recall系数
            recall.task = task
            recall.original = original_recall
            recall.cfs = cfs_recall
            recall.alpha = alp_recall
            recall.save()
            f1score = F1score()  # 保存f1score系数
            f1score.task = task
            f1score.original = original_f1
            f1score.cfs = cfs_f1
            f1score.alpha = alp_f1
            f1score.save()

            taskresult.task = task
            taskresult.original_accuracy = original_accuracy
            taskresult.cfs_result = backend.list2str(cfs_ans.tolist())
            taskresult.cfs_accuracy = cfs_accuracy  # 准确度按百分比显示，所以乘100
            taskresult.cfs_time = cfs_time
            taskresult.alphainvesting_result = backend.list2str(alp_ans.tolist())
            taskresult.alphainvesting_accuracy = alp_accuracy
            taskresult.alphainvesting_time = alp_time
            taskresult.rlf_time = rlf_time

            dictionary = {}  # 为选择最佳特征子集做准备
            dictionary1 = {}  # CFS算法的结果
            dictionary2 = {}  # alpha算法的结果
            dictionary3 = {}  # chi算法的结果
            dictionary4 = {}  # ReliefF算法的结果

            list_cfs = [cfs_accuracy / 100, cfs_recall / 100, cfs_f1 / 100]
            list_alpha = [alp_accuracy / 100, alp_recall / 100, alp_f1 / 100]
            dictionary1[taskresult.cfs_result] = list_cfs
            dictionary2[taskresult.alphainvesting_result] = list_alpha
            out = open(user.chians_address + chi_file, 'rb')
            chi_file = np.loadtxt(out, dtype=str, delimiter=',', skiprows=0)
            a, b = chi_file.shape
            for i in range(a):
                dictionary3[chi_file[i, 0]] = [float(chi_file[i, 1]) / 100, float(chi_file[i, 3]) / 100, float(chi_file[i, 4]) / 100]
            rlf_file = ReliefFResult.objects.filter(task_id=task_id)
            for i in rlf_file:
                dictionary4[i.rlf_result] = [i.rlf_accuracy / 100, i.rlf_recall / 100, i.rlf_f1 / 100]
            dictionary['cfs'] = dictionary1
            dictionary['alp'] = dictionary2
            dictionary['chi'] = dictionary3
            dictionary['rlf'] = dictionary4
            best_set = backend.select_the_bestset(dictionary)
            taskresult.best_result_system = best_set[1][0]
            taskresult.system_result_from = best_set[0]
            taskresult.save()
            return HttpResponseRedirect(reverse('featureselection:show_result', args=(task_id,)))
    except BaseException:  # 若中间某个位置发生过错误，则删除任何一个该文件的记录
        form = TaskForm()
        form['feature_names'] = task.feature_names
        form['label_name'] = task.label_name
        form.add_error(None, ValidationError('There Is An Error In The File. Please Check And Try Again.'))
        try:
            f1_ = F1score.objects.get(task_id=task_id)
            f1_.delete()
        except BaseException:
            pass
        try:
            recall_ = Recall.objects.get(task_id=task_id)
            recall_.delete()
        except BaseException:
            pass
        try:
            user_ = UsersResult.objects.filter(task_id=task_id)
            user_.delete()
        except BaseException:
            pass
        try:
            rlf_ = ReliefFResult.objects.filter(task_id=task_id)
            rlf_.delete()
        except BaseException:
            pass
        try:
            taskresult_ = TaskResult.objects.get(task_id=task_id)
            chi_ = taskresult_.chi_file
            taskresult.delete()
        except BaseException:
            pass
        try:
            os.remove(user.chians_address + chi_)
        except BaseException:
            pass
        try:
            task_ = Task.objects.get(task_id=task_id)
            task_.delete()
        except BaseException:
            pass
        try:
            os.remove(a)  # 删除上传的文件
        except BaseException:
            pass
        return render(request, 'featureselection/task.html', {'form': form})  # 将错误信息显示到前端
    else:
        pass
    return render(request, 'featureselection/task_result.html')


@login_required
def show_result(request, task_id):
    result = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    features_list = backend.str2list(task.feature_names)
    cid = {}
    fnames = backend.str2list(task.feature_names)
    for i in range(len(fnames)):
        cid[i] = fnames[i]
    try:
        algorithm_echarts = backend.algorithm_bar(result, task_id).render_embed()
        try:
            line, chi_ans = backend.chi_line(task_id)
            chi_es = line.render_embed()
            chi_ans_dic = backend.np2dic(chi_ans)
            sum1, pie_es = backend.statistics_pie_chart()
            pie_es = pie_es.render_embed()
        except BaseException:
            chi_es = None
            chi_ans_dic = None
        try:
            rlf_set = ReliefFResult.objects.filter(task_id=task_id)
            rlf_line = backend.rlf_line(rlf_set)
            rlf_echarts = rlf_line.render_embed()
        except BaseException:
            rlf_set = []
            rlf_echarts = None
    except BaseException:
        pass
    else:
        pass
    try:  # 获得算法的recall值
        recall = Recall.objects.get(task_id=task_id)
    except BaseException:
        recall = []
    try:  # 获得算法的f1值
        f1score = F1score.objects.get(task_id=task_id)
    except BaseException:
        f1score = []
    if task.task_type == 'classification':
        return render(request, 'featureselection/classification_task_result.html', {
            'result': result,
            'features_list': features_list,
            'task_id': task_id,
            'algorithm_echarts': algorithm_echarts,
            'cid': cid,
            'chi_es': chi_es,
            'chi_ans_dic': chi_ans_dic,
            'rlf_set': rlf_set,
            'rlf_echarts': rlf_echarts,
            'recall': recall,
            'f1score': f1score,
            'sum1': sum1,
            'pie_es': pie_es})
    else:
        pass


# 展示用户自选页面
@login_required
def show_user_result(request, task_id):
    result = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    features_list = backend.str2list(task.feature_names)
    cid = {}
    fnames = backend.str2list(task.feature_names)
    for i in range(len(fnames)):
        cid[i] = fnames[i]
    try:
        userresult_list = UsersResult.objects.filter(task_id=task_id)
        if len(userresult_list) == 0:
            user_echarts = None
        else:
            user_echarts = backend.echarts_bar(userresult_list).render_embed()
    except BaseException:
        userresult_list = []
        user_echarts = None
    try:
        algorithm_echarts = backend.algorithm_bar(result, task_id).render_embed()
        try:
            line, chi_ans = backend.chi_line(task_id)
            chi_es = line.render_embed()
            chi_ans_dic = backend.np2dic(chi_ans)
            sum1, pie_es = backend.statistics_pie_chart()
            pie_es = pie_es.render_embed()
        except BaseException:
            chi_es = None
            chi_ans_dic = None
        try:
            rlf_set = ReliefFResult.objects.filter(task_id=task_id)
            rlf_line = backend.rlf_line(rlf_set)
            rlf_echarts = rlf_line.render_embed()
        except BaseException:
            rlf_set = []
            rlf_echarts = None
    except BaseException:
        pass
    return render(request, 'featureselection/userchoose.html', {
        'task_id': task_id,
        'features_list': features_list,
        'cid': cid,
        'userresult_list': userresult_list,
        'user_echarts': user_echarts,
        'algorithm_echarts': algorithm_echarts,
        'chi_es': chi_es,
        'rlf_echarts': rlf_echarts,
    })


# 展示用户自选页面
@login_required
def show_best_result(request, task_id):
    task = Task.objects.get(id=task_id)
    result = TaskResult.objects.get(task_id=task_id)
    cid = {}
    fnames = backend.str2list(task.feature_names)
    for i in range(len(fnames)):
        cid[i] = fnames[i]
    return render(request, 'featureselection/bestfeaturesubset.html', {
        'cid': cid,
        'result': result,
        'task_id': task_id
    })


#  未完善  添加其他算法需要修改
@login_required
def export_result(request, task_id, index1, index2):
    taskresult = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    user = MyUser.objects.get(username=task.user)
    myfile = user.upload_address+str(task.upload_file)
    if index1 == 0:   # 将用户自选的导出
        export = user.export_address + str(task_id) + '_userown'+'_export_' + str(index2) + '_' + str(task.upload_file)
        obj = UsersResult.objects.get(id=index2)
        usersresult = obj.user_result
        obj.exportfile = str(task_id) + '_userown'+'_export_' + str(index2) + '_' + str(task.upload_file)
        obj.save()
        fs_result = backend.str2list(usersresult)
    elif index1 == 1:
        export = user.export_address + str(task_id) + '_cfs_export_' + str(task.upload_file)
        fs_result = backend.str2list(taskresult.cfs_result)
    elif index1 == 2:
        export = user.export_address + str(task_id) + '_alp_export_' + str(task.upload_file)
        fs_result = backend.str2list(taskresult.alphainvesting_result)
    elif index1 == 3:
        export = user.export_address + str(task_id) + '_chi_export_' + str(index2) + '_' + str(task.upload_file)
        if not os.path.exists(export):
            backend.chi_result(task_id, index2, False,  str(task_id) + '_chi_export_' + str(index2) + '_' + str(task.upload_file))
            fs_result = backend.chi_result(task_id, index2, True, export)
    elif index1 == 4:
        export = user.export_address + str(task_id) + '_rlf_export_' + str(index2) + '_' + str(task.upload_file)
        rlf_result = ReliefFResult.objects.get(id=index2)
        if not os.path.exists(export):
            rlf_result.rlf_export = str(task_id) + '_rlf_export_' + str(index2) + '_' + str(task.upload_file)
            rlf_result.save()
            fs_result = backend.str2list(rlf_result.rlf_result)
    else:
        pass
    if not os.path.exists(export):
        fs_result = np.array(fs_result)
        data, label = backend.simple_load_csv(myfile)
        data = backend.change_datashape(data, fs_result)
        data = np.vstack((data.T, label)).T
        np.savetxt(export, data, fmt='%s', delimiter=',')
    else:
        '已经存在了这个文件'
    return HttpResponseRedirect(reverse('featureselection:show_result', args=(task_id,)))


@login_required
def download(request, task_id, index1, index2):
    def file_iterator(file, chunk_size=512):
        with open(file) as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break
    taskresult = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    user = MyUser.objects.get(username=task.user)
    myfile = user.upload_address + str(task.upload_file)
    if index1 == 0:  # 将用户自选的导出
        export = user.export_address + str(task_id) + '_userown' + '_export_' + str(index2) + '_' + str(
            task.upload_file)
        obj = UsersResult.objects.get(id=index2)
        usersresult = obj.user_result
        obj.exportfile = str(task_id) + '_userown' + '_export_' + str(index2) + '_' + str(task.upload_file)
        obj.save()
        fs_result = backend.str2list(usersresult)
    elif index1 == 1:
        export = user.export_address + str(task_id) + '_cfs_export_' + str(task.upload_file)
        fs_result = backend.str2list(taskresult.cfs_result)
    elif index1 == 2:
        export = user.export_address + str(task_id) + '_alp_export_' + str(task.upload_file)
        fs_result = backend.str2list(taskresult.alphainvesting_result)
    elif index1 == 3:
        export = user.export_address + str(task_id) + '_chi_export_' + str(index2) + '_' + str(task.upload_file)
        if not os.path.exists(export):
            backend.chi_result(task_id, index2, False,
                               str(task_id) + '_chi_export_' + str(index2) + '_' + str(task.upload_file))
            fs_result = backend.chi_result(task_id, index2, True, export)
    elif index1 == 4:
        export = user.export_address + str(task_id) + '_rlf_export_' + str(index2) + '_' + str(task.upload_file)
        rlf_result = ReliefFResult.objects.get(id=index2)
        if not os.path.exists(export):
            rlf_result.rlf_export = str(task_id) + '_rlf_export_' + str(index2) + '_' + str(task.upload_file)
            rlf_result.save()
            fs_result = backend.str2list(rlf_result.rlf_result)
    else:
        pass
    if not os.path.exists(export):
        fs_result = np.array(fs_result)
        data, label = backend.simple_load_csv(myfile)
        data = backend.change_datashape(data, fs_result)
        data = np.vstack((data.T, label)).T
        np.savetxt(export, data, fmt='%s', delimiter=',')
    else:
        pass
    file = export
    output_file = 'export.csv'
    response = StreamingHttpResponse(file_iterator(file))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(output_file)
    return response


@login_required
def download_bestsubset(request, task_id):
    def file_iterator(file):
        with open(file) as f:
            while True:
                c = f.readline()
                if c:
                    yield c
                else:
                    break
    taskresult = TaskResult.objects.get(task_id=task_id)
    task = Task.objects.get(id=task_id)
    user = MyUser.objects.get(username=task.user)
    myfile = user.upload_address + str(task.upload_file)
    fs_result = backend.str2list(taskresult.best_result_system)
    export = user.export_address + 'best_result_' + str(task_id) + '.csv'
    if not os.path.exists(export):
        fs_result = np.array(fs_result)
        data, label = backend.simple_load_csv(myfile)
        data = backend.change_datashape(data, fs_result)
        data = np.vstack((data.T, label)).T
        np.savetxt(export, data, fmt='%s', delimiter=',', newline='\n')
    file = export
    output_file = 'export.csv'
    response = StreamingHttpResponse(file_iterator(file))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(output_file)
    return response


@login_required
def analyze_user_choice(request, task_id):
    feature_list = request.POST.getlist('features_checkbox')
    if len(feature_list) != 0:
        task = Task.objects.get(id=task_id)
        user = MyUser.objects.get(username=task.user)
        feature_names = backend.str2list(task.feature_names)
        index1 = []
        for f in feature_list:
            index1.append(feature_names.index(f))
        index1 = np.array(index1)
        data, labels, dict = backend.load_csv(user.upload_address+str(task.upload_file))
        data = backend.change_datashape(data, index1)
        accuracy, recall, f1 = backend.naviebayes(data, labels)
        accuracy = round(accuracy * 100, 2)
        recall = round(recall * 100, 2)
        f1 = round(f1 * 100, 2)
        userresult = UsersResult()
        userresult.task = task
        userresult.user_result = backend.list2str(index1.tolist())
        userresult.accuracy = accuracy
        userresult.recall = recall
        userresult.f1 = f1
        userresult.save()
    else:
        'no checkbox was choice'  # --------------------------------
    return HttpResponseRedirect(reverse('featureselection:show_user_result', args=(task_id,)))


@login_required
def delete_own_result(request, r_id):
    usersresult = UsersResult.objects.get(id=r_id)
    task_id = usersresult.task.id
    usersresult.delete()
    return HttpResponseRedirect(reverse('featureselection:show_user_result', args=(task_id,)))


@login_required
def show_history(request, username):
    try:
        task_set = Task.objects.filter(user_id=username).order_by('-date_added')
    except BaseException:
        task_set = []
    return render(request, 'featureselection/history.html', {'task_set': task_set})


@login_required
def delete_task(request, task_id, username):
    user = MyUser.objects.get(username=username)
    up_add = user.upload_address
    ep_add = user.export_address
    ci_add = user.chians_address
    try:
        rlf_result = ReliefFResult.objects.filter(task_id=task_id)
        for v in rlf_result:
            if v.rlf_export != 'NotExport':
                print(ep_add + v.rlf_export)
                os.remove(ep_add + v.rlf_export)
        rlf_result.delete()
    except BaseException:
        pass
    try:    # 删除chi算法导出文件,必须在其他之前
        task_reuslt = TaskResult.objects.get(task_id=task_id)
        user = MyUser.objects.get(username=username)
        chi_file_address = user.chians_address
        out = open(chi_file_address + task_reuslt.chi_file, 'rb')
        file = np.loadtxt(out, delimiter=',', dtype=str, skiprows=0)
        out.close()
        file = file[:, -1].tolist()
        for i in file:
            if i != 'NotExport':
                os.remove(ep_add + i)
    except BaseException:
        pass
    try:
        userresult = UsersResult.objects.filter(task_id=task_id)
        for ur in userresult:
            if ur.exportfile != 'True':
                os.remove(ep_add + ur.exportfile)
        userresult.delete()
        try:
            taskresult = TaskResult.objects.get(task_id=task_id)
            cfile = taskresult.chi_file
            os.remove(ci_add + cfile)
            taskresult.delete()
            try:
                task = Task.objects.get(id=task_id)
                uf = task.upload_file
                task.delete()
                os.remove(up_add + str(uf))
                os.remove(ep_add + str(task_id) + '_cfs_export_' + str(uf))
                os.remove(ep_add + str(task_id) + '_alp_export_' + str(uf))
            except BaseException:
                pass
        except BaseException:
            pass
    except BaseException:
        pass
    return HttpResponseRedirect(reverse('featureselection:show_history', args=(username,)))


@login_required
def choose_result(request, task_id, index1, index2):
    task_result = TaskResult.objects.get(task_id=task_id)
    statistics = Statistics.objects.get(id=1)
    if index1 == 0:
        statistics.User_num += 1
        task_result.choosed_result_from = 'user'
        task_result.choosed_result = UsersResult.objects.get(id=index2).user_result
    if index1 == 1:
        statistics.CFS_num += 1
        task_result.choosed_result_from = 'cfs'
        task_result.choosed_result = task_result.cfs_result
    if index1 == 2:
        statistics.Alpha_num += 1
        task_result.choosed_result_from = 'alp'
        task_result.choosed_result = task_result.alphainvesting_result
    if index1 == 3:
        statistics.Chi2_num += 1
        task_result.choosed_result_from = 'chi'
        task = Task.objects.get(id=task_id)
        chiadd = MyUser.objects.get(username=task.user).chians_address
        out = open(chiadd + task_result.chi_file, 'r')
        file = np.loadtxt(out, delimiter=',', dtype=str, skiprows=0)
        out.close()
        task_result.choosed_result = file[index2, 0]
    if index1 == 4:
        statistics.ReliefF_num += 1
        task_result.choosed_result_from = 'rlf'
        task_result.choosed_result = ReliefFResult.objects.get(id=index2).rlf_result
    statistics.save()
    task_result.ischoosed = True
    task_result.save()
    if index1 == 0:
        return HttpResponseRedirect(reverse('featureselection:show_user_result', args=(task_id,)))
    else:
        return HttpResponseRedirect(reverse('featureselection:show_result', args=(task_id,)))

