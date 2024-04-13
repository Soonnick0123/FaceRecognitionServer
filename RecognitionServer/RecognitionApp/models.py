from django.db import models
from django.utils import timezone
import os

def customer_directory_path(instance, filename):
    customer_username = ''.join(x for x in instance.username if x.isalnum())
    return os.path.join('customer_photos', customer_username, filename)

class Customer(models.Model):
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    name = models.CharField(max_length=255, verbose_name="Name",null=False)
    username = models.CharField(max_length=255, verbose_name="Username",null=False)
    email = models.EmailField(verbose_name="Email Address",null=False)
    phone = models.CharField(max_length=20, verbose_name="Phone Number",null=False)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="Gender",null=False)
    # photo = models.ImageField(upload_to=customer_directory_path, verbose_name="Photo", blank=False, null=False)
    photo1 = models.ImageField(upload_to=customer_directory_path, verbose_name="Photo 1", blank=False, null=False)
    photo2 = models.ImageField(upload_to=customer_directory_path, verbose_name="Photo 2", blank=True, null=True)
    photo3 = models.ImageField(upload_to=customer_directory_path, verbose_name="Photo 3", blank=True, null=True)

    def __str__(self):
        return self.name

class LoginRecord(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='login_records')
    login_time = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.customer.name} logged in at {self.login_time}"