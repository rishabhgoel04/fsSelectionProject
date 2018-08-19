from .models import Task
from django import forms
from django.forms import TextInput
import featureselection.backend as backend
import os
from featureselection.models import MyUser
# -----------------------


class TaskForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ['feature_names', 'label_name', 'upload_file']
        labels = {'feature_names': '特征名称', 'label_name': '标签名称', 'upload_file': '上传文件'}
        widgets = {'feature_names': TextInput(attrs={'placeholder': "特征名称（以逗号隔开）"})}
        help_texts = {
            'upload_file': "*支持csv、txt格式文件，数据以逗号分隔",
        }

    def save(self, user, filename, label_values, commit=True):
        # Save the provided password in hashed format
        taskform = super().save(commit=False)
        taskform.user = user
        taskform.upload_file = filename
        taskform.label_values = label_values
        taskform.task_type = 'classification'
        if commit:
            taskform.save()
        return taskform

    def clean(self):
        user = MyUser.objects.get(username=self.user)
        feature_names = self.cleaned_data['feature_names']
        try:
            file = self.cleaned_data['upload_file']
        except BaseException:
            raise forms.ValidationError('提交错误：空的文件。')
        if os.path.splitext(file.name)[1] != '.csv' and os.path.splitext(file.name)[1] != '.txt':
            raise forms.ValidationError('提交错误：确保文件格式是txt或者csv。')
        if os.path.exists(user.upload_address + file.name):
            raise forms.ValidationError('提交错误：您已提交了有相同文件名的数据集，请修改文件名后重试。')
        column_num = backend.get_column_num(file, 1)
        num = len(backend.str2list(feature_names))
        if num != (column_num - 1):
            raise forms.ValidationError('提交错误：请确保特征数与数据集中保持一致。')
        return self.cleaned_data

