from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .models import Book

class BookHomeView(ListView):
    model = Book
    template_name = 'library/home.html'
    context_object_name = 'books'
    ordering = ['-title']