from django import forms
from .models import BookComments

class CommentCreateForm(forms.ModelForm):

    class Meta:
        model = BookComments
        fields = ['content']