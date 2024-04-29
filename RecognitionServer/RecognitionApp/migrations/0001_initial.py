# Generated by Django 4.2.10 on 2024-04-04 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, verbose_name='Name')),
                ('email', models.EmailField(max_length=254, verbose_name='Email Address')),
                ('phone', models.CharField(max_length=20, verbose_name='Phone Number')),
                ('gender', models.CharField(choices=[('M', 'Male'), ('F', 'Female')], max_length=1, verbose_name='Gender')),
                ('photo', models.ImageField(blank=True, null=True, upload_to='customer_photos/', verbose_name='Photo')),
            ],
        ),
    ]
