# Generated by Django 2.0.5 on 2018-05-15 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('featureselection', '0008_usersresult'),
    ]

    operations = [
        migrations.AddField(
            model_name='usersresult',
            name='exportfile',
            field=models.CharField(default=True, max_length=200),
        ),
    ]