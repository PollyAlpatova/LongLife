
from django.shortcuts import render


def index(request):
    return render(request,'server/temperature.html')

# Create your views here.
