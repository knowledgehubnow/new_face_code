# Generated by Django 4.2.6 on 2023-11-09 07:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0026_videorecognition_appropriate_facial'),
    ]

    operations = [
        migrations.AddField(
            model_name='videorecognition',
            name='body_posture',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
