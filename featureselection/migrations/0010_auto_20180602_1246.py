# Generated by Django 2.0.5 on 2018-06-02 04:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('featureselection', '0009_usersresult_exportfile'),
    ]

    operations = [
        migrations.CreateModel(
            name='F1score',
            fields=[
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='featureselection.Task')),
                ('original', models.FloatField()),
                ('alpha', models.FloatField()),
                ('cfs', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Recall',
            fields=[
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='featureselection.Task')),
                ('original', models.FloatField()),
                ('alpha', models.FloatField()),
                ('cfs', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='ReliefFResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rlf_result', models.CharField(max_length=1000)),
                ('rlf_accuracy', models.FloatField()),
                ('rlf_export', models.CharField(default=True, max_length=1000)),
                ('rlf_recall', models.FloatField(default=True)),
                ('rlf_f1', models.FloatField(default=True)),
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='featureselection.Task')),
            ],
        ),
        migrations.AddField(
            model_name='taskresult',
            name='rlf_time',
            field=models.FloatField(default=True),
        ),
        migrations.AddField(
            model_name='usersresult',
            name='f1',
            field=models.FloatField(default=True),
        ),
        migrations.AddField(
            model_name='usersresult',
            name='recall',
            field=models.FloatField(default=True),
        ),
    ]
