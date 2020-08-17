from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email']


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['image']

class ProfileCreditForm(forms.Form):
    CHOICES = (('0', 0),
            ('5', 5),
            ('10', 10),
            ('15', 15),
            ('25', 25),)
    credit = forms.ChoiceField(widget=forms.Select(),
                                         choices=CHOICES, required=False)
    