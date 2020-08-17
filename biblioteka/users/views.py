from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm, ProfileCreditForm
from django.views.generic import DetailView, ListView
from .models import Profile


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Your account has been created! You are now able to log in')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST,
                                   request.FILES,
                                   instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render(request, 'users/profile.html', context)

@login_required
def add_credit(request):
    if request.method == 'POST':
        c_form = ProfileCreditForm(request.POST)
        if c_form.is_valid():
            credit = float(c_form.cleaned_data.get('credit'))
            user_credit = request.user.profile.credit
            user_credit += credit
            request.user.profile.credit = user_credit
            request.user.profile.save()
            messages.success(request, f'Credit has been added successfully!')
            return redirect('profile')
    else:
        c_form = ProfileCreditForm()
    return render(request, 'users/add_credit.html', {'c_form': c_form})

@login_required
def my_books(request):
    user_profile = Profile.objects.filter(user=request.user)
    if user_profile.exists():
        u_profile = user_profile[0]
        user_books = u_profile.books.all()
        context = {'user_books': user_books}

    return render(request, 'users/my_books.html', context)