
from django.urls import path
form . import views


urlpatterns = [
    path('/', views.HomeList.as_view(), name='library-home'),
]
