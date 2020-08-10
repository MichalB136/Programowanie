
from django.urls import path
from . import views


urlpatterns = [
    path('', views.BookHomeView.as_view(), name='library-home'),
    path('book/new/', views.BookCreateView.as_view(), name='library-create'),
    path('book/<slug>/', views.BookDetailView.as_view(), name='library-detail'),
    path('book/<slug>/update/', views.BookUpdateView.as_view(), name='library-update'),
    path('book/<slug>/delete/', views.BookDeleteView.as_view(), name='library-delete'),

]
