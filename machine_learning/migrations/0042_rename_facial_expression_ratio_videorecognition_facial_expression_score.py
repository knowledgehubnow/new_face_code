# Generated by Django 4.2.6 on 2023-12-20 04:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0041_videorecognition_facial_expression_ratio'),
    ]

    operations = [
        migrations.RenameField(
            model_name='videorecognition',
            old_name='facial_expression_ratio',
            new_name='facial_expression_score',
        ),
    ]
