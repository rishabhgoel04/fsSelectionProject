from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.auth.views import logout
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth import login, logout, authenticate
#  from django.contrib.auth.forms import UserCreationForm
from .admin import UserCreationForm
from django.contrib.auth import get_user_model
User = get_user_model()
# Create your views here.


@login_required
def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse('featureselection:index'))


def register(request):
    if request.method != 'POST':
        form = UserCreationForm()
    else:
        form = UserCreationForm(data=request.POST)
        if form.is_valid():
            new_user = form.save()
            authenticated_user = authenticate(username=new_user.username, password=request.POST['password1'])
            login(request, authenticated_user)
            return HttpResponseRedirect(reverse('featureselection:index'))
    context = {'form': form}
    return render(request, 'users/register.html', context)
