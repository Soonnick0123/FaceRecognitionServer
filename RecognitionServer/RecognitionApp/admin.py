from django.contrib import admin
from .models import *

class CustomerAdmin(admin.ModelAdmin):
    list_display = ("id", 'username', 'name', 'email', 'phone', 'gender')
    search_fields = ('name', 'email')

admin.site.register(Customer, CustomerAdmin)