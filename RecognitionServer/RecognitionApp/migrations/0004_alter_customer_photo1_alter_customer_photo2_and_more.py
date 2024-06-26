# Generated by Django 4.2.10 on 2024-04-12 08:58

import RecognitionApp.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RecognitionApp', '0003_remove_customer_photo_customer_photo1_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='customer',
            name='photo1',
            field=models.ImageField(null=True, upload_to=RecognitionApp.models.customer_directory_path, verbose_name='Photo 1'),
        ),
        migrations.AlterField(
            model_name='customer',
            name='photo2',
            field=models.ImageField(blank=True, null=True, upload_to=RecognitionApp.models.customer_directory_path, verbose_name='Photo 2'),
        ),
        migrations.AlterField(
            model_name='customer',
            name='photo3',
            field=models.ImageField(blank=True, null=True, upload_to=RecognitionApp.models.customer_directory_path, verbose_name='Photo 3'),
        ),
    ]
