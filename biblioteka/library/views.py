from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .models import Book

class BookHomeView(ListView):
    model = Book
    template_name = 'library/home.html'
    context_object_name = 'books'
    ordering = ['title']

class BookDetailView(DetailView):
    model = Book
    template_name = 'library/detail.html'

class BookUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Book
    template_name = 'library/update_book.html'
    fields = ['author','title', 'genre', 'description','image']

    def test_func(self):
        if self.request.user.is_staff:
            return True
        return False

    
    def form_valid(self, form):
        messages.success(self.request, f'Book has been updated')
        return super().form_valid(form)

class BookCreateView(LoginRequiredMixin, UserPassesTestMixin, CreateView):
    model = Book
    template_name = 'library/create_book.html'
    fields = ['author','title', 'genre', 'description','image']

    def test_func(self):
        if self.request.user.is_staff:
            return True
        return False

    def form_valid(self, form):
        messages.success(self.request, f'Book has been created')
        return super().form_valid(form)

class BookDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Book
    template_name = 'library/delete_book.html'
    success_url = '/'

    def test_func(self):
        if self.request.user.is_staff:
            return True
        return False