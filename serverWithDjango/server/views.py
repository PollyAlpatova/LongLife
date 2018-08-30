
from django.http import render


def index(request):
    return render(request,"Hello, World!")

# Create your views here.
