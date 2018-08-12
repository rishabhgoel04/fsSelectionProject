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
        widgets = {'feature_names': TextInput(attrs={'placeholder': "Feature names And Separated with ','"})}
        help_texts = {
            'upload_file': "*Only supports csv, txt format files.And the delimiter in File is ','",
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
            raise forms.ValidationError('Empty File!')
        if os.path.splitext(file.name)[1] != '.csv' and os.path.splitext(file.name)[1] != '.txt':
            raise forms.ValidationError('Please Ensure The File Format CSV Or TXT')
        if os.path.exists(user.upload_address + file.name):
            raise forms.ValidationError('There Is Already existed The File.Please Change The Name And Try again.')
        column_num = backend.get_column_num(file, 1)
        num = len(backend.str2list(feature_names))
        if num != (column_num - 1):
            raise forms.ValidationError('Please Ensure The Number Of Features Same To The File.')
        return self.cleaned_data

