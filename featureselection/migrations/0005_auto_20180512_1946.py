# Generated by Django 2.0.5 on 2018-05-12 11:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('featureselection', '0004_taskresult_chi_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='taskresult',
            name='task',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='featureselection.Task', unique=True),
        ),
    ]