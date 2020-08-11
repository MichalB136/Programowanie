from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .models import Book, Author, Genre, Order, OrderBook

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

class AuthorCreateView(LoginRequiredMixin, UserPassesTestMixin, CreateView):
    model = Author
    template_name = 'library/create_author.html'
    fields = ['first_name', 'last_name']

    def test_func(self):
        if self.request.user.is_staff:
            return True
        return False

    def form_valid(self, form):
        messages.success(self.request, f'Author has been created')
        return super().form_valid(form)

class GenreCreateView(LoginRequiredMixin, UserPassesTestMixin, CreateView):
    model = Genre
    template_name = 'library/create_genre.html'
    fields = ['genre']

    def test_func(self):
        if self.request.user.is_staff:
            return True
        return False

    def form_valid(self, form):
        messages.success(self.request, f'Author has been created')
        return super().form_valid(form)

def add_to_cart(request, slug):
    book = get_object_or_404(Book, slug=slug)
    order_book, created = OrderBook.objects.get_or_create(book=book)
    order_qs = Order.objects.filter(user=request.user, ordered=False)
    if order_qs.exists():
        order = order_qs[0]
        if not order.books.filter(book__slug=book.slug).exists():
            order.books.add(order_book)
            messages.success(request, f'Order has been created!')
        else:
            messages.warning(request, f'Order already exists!')
    else:
        order = Order.objects.create(user=request.user)
        order.books.add(order_book)
    return redirect('library-detail', slug= slug)
    
def remove_from_cart(request, slug):
    book = get_object_or_404(Book, slug=slug)
    order_qs = Order.objects.filter(user=request.user, ordered=False)
    if order_qs.exists():
        order = order_qs[0]
        if order.books.filter(book__slug=book.slug).exists():
            order_book = OrderBook.objects.filter(book=book)[0]
            order.books.remove(order_book)
            messages.success(request, f'Order has been removed!')
        else:
            messages.warning(request, f'Order does not containt this book!')
            return redirect('cart')
    else:
        messages.warning(request, f"You don't have an order!")
        return redirect('cart')
    return redirect('cart')

class CartListView(ListView):
    model = Order
    template_name = 'library/cart.html'
    context_object_name = 'orders'