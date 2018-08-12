from django.db import models
from users.models import MyUser
# Create your models here.


class Task(models.Model):
    user = models.ForeignKey(MyUser, on_delete=models.CASCADE)
    feature_names = models.CharField(max_length=1000)  # 特征的名称
    label_name = models.CharField(max_length=100)  # 标签的名称
    label_values = models.CharField(max_length=300)  # 标签的值
    t_type = (('classification', 'classification'), ('regression', 'regression'))  # 本特征选择的任务的类型
    task_type = models.CharField(choices=t_type, max_length=100)
    upload_file = models.FileField()
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.feature_names


# 在featureselection views中task_result中有使用
class TaskResult(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, primary_key=True, unique=True)
    original_accuracy = models.CharField(max_length=1000)
    cfs_result = models.CharField(max_length=1000)
    cfs_accuracy = models.FloatField()
    cfs_time = models.FloatField(default=True)
    alphainvesting_result = models.CharField(max_length=1000)
    alphainvesting_accuracy = models.FloatField()
    alphainvesting_time = models.FloatField(default=True)
    chi_result = models.CharField(max_length=1000)
    chi_file = models.CharField(max_length=1000)
    chi_time = models.FloatField(default=True)
    rlf_time = models.FloatField(default=True)
    best_result_system = models.CharField(max_length=1000, default=True)  # 系统推荐的最佳特征子集
    system_result_from = models.CharField(max_length=1000, default=True)
    ischoosed = models.BooleanField(default=False)
    choosed_result = models.CharField(max_length=1000, default=True)  # 用户最终选择的特征子集
    choosed_result_from = models.CharField(max_length=1000, default=True)  # 被选择的结果来自哪个算法


    def __str__(self):
        return self.task


class ReliefFResult(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    rlf_result = models.CharField(max_length=1000)
    rlf_accuracy = models.FloatField()
    rlf_export = models.CharField(max_length=1000, default=True)
    rlf_recall = models.FloatField(default=True)
    rlf_f1 = models.FloatField(default=True)


class UsersResult(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    user_result = models.CharField(max_length=100)
    accuracy = models.FloatField()
    exportfile = models.CharField(max_length=200, default=True)
    recall = models.FloatField(default=True)
    f1 = models.FloatField(default=True)


class Recall(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, primary_key=True)
    original = models.FloatField()
    alpha = models.FloatField()
    cfs = models.FloatField()


class F1score(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, primary_key=True)
    original = models.FloatField()
    alpha = models.FloatField()
    cfs = models.FloatField()


class Statistics(models.Model):  # 统计特征选择结果的最终被选择的情况，便于后期统计与改进。
    CFS_num = models.IntegerField(default=0)
    Alpha_num = models.IntegerField(default=0)
    Chi2_num = models.IntegerField(default=0)
    ReliefF_num = models.IntegerField(default=0)
    User_num = models.IntegerField(default=0)

