from django.contrib import admin
from .models import *

class CustomerAdmin(admin.ModelAdmin):
    list_display = ("id", 'username', 'name', 'email', 'phone', 'gender')
    search_fields = ('name', 'email')

class RecordAdmin(admin.ModelAdmin):
    list_display = ('customer','login_time')

admin.site.register(Customer, CustomerAdmin)
admin.site.register(LoginRecord, RecordAdmin)