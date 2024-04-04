from django.db import models


class Customer(models.Model):
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    name = models.CharField(max_length=255, verbose_name="Name")
    email = models.EmailField(verbose_name="Email Address")
    phone = models.CharField(max_length=20, verbose_name="Phone Number")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="Gender")
    photo = models.ImageField(upload_to='customer_photos/', verbose_name="Photo", null=True, blank=True)

    def __str__(self):
        return self.name
