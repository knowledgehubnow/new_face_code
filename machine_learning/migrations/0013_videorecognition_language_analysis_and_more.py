# Generated by Django 4.2.6 on 2023-10-16 11:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0012_analyzepdf_pdf_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='videorecognition',
            name='language_analysis',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='videorecognition',
            name='voice_modulation_analysis',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
