
from django.urls import path
from . import views


urlpatterns = [
    path('', views.BookHomeView.as_view(), name='library-home'),
]
