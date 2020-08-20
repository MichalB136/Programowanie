from django.urls import path
from . import views


urlpatterns = [
    path('', views.BookHomeView.as_view(), name='library-home'),
    path('book/new/', views.BookCreateView.as_view(), name='library-create'),
    path('book/<slug>/', views.BookDetailView.as_view(), name='library-detail'),
    path('book/<slug>/new-comment', views.CommentCreateView.as_view(), name='library-new-comment'),
    path('book/<slug>/update/', views.BookUpdateView.as_view(), name='library-update'),
    path('book/<slug>/delete/', views.BookDeleteView.as_view(), name='library-delete'),
    path('author/new/', views.AuthorCreateView.as_view(), name='author-create'),
    path('genre/new/', views.GenreCreateView.as_view(), name='genre-create'),
    path('add-to-cart/<slug>/', views.add_to_cart, name='add-to-cart'),
    path('remove-from-cart/<slug>/', views.remove_from_cart, name='remove-from-cart'),
    path('cart/', views.CartListView.as_view(), name='cart'),
    path('add-to-account/', views.add_to_account, name='add-to-account'),
]
