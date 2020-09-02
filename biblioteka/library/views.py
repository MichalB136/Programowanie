from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q
from django.utils import timezone
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from operator import attrgetter
from .models import Book, Author, Genre, Order, OrderBook, BookComments
from .forms import CommentCreateForm

class BookHomeView(ListView):
    model = Book
    template_name = 'library/home.html'
    context_object_name = 'books'
    ordering = ['title']

    def get_queryset(self):
        context = {}
        query = ""
        if self.request.GET:
            query = self.request.GET['search']
            context['query'] = str(query)
            search_list = sorted(get_books_queryset(query), key=attrgetter('title'))
            return search_list
        return super().get_queryset()
    

def get_books_queryset(query=None):
    queryset = []
    queries = query.split(" ")
    for q in queries:
        books = Book.objects.filter(title__icontains=q).distinct()
        for book in books:
            queryset.append(book)
        return list(set(queryset))


# class BookDetailView(DetailView):
#     model = Book
#     template_name = 'library/detail.html'

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

class CartListView(LoginRequiredMixin, ListView):
    model = Order
    template_name = 'library/cart.html'
    context_object_name = 'orders'

@login_required
def add_to_cart(request, slug):
    logged_user = request.user
    book = get_object_or_404(Book, slug=slug)
    order_book, created = OrderBook.objects.get_or_create(user = logged_user, book=book)
    order_qs = Order.objects.filter(user=logged_user, ordered=False)
    if book in logged_user.profile.books.all():
        messages.warning(request, f'You already have that book')
    else:
        if order_qs.exists():
            order = order_qs[0]
            if not order.books.filter(book__slug=book.slug).exists():
                order.books.add(order_book)
                messages.success(request, f'Order has been created!')
            else:
                messages.warning(request, f'Order already exists!')
        else:
            ordered_date = timezone.now()
            order = Order.objects.create(user=logged_user, ordered_date=ordered_date)
            order.books.add(order_book)
    return redirect('library-detail', slug= slug)

@login_required    
def remove_from_cart(request, slug):
    book = get_object_or_404(Book, slug=slug)
    order_qs = Order.objects.filter(user=request.user, ordered=False)
    if order_qs.exists():
        order = order_qs[0]
        if order.books.filter(book__slug=book.slug).exists():
            order_book = OrderBook.objects.filter(book=book)[0]
            order.books.remove(order_book)
            order_book.delete()
            if order.books.count() == 0:
                order.delete()
            messages.success(request, f'Order has been removed!')
        else:
            messages.warning(request, f'Order does not containt this book!')
            return redirect('cart')
    else:
        messages.warning(request, f"You don't have an order!")
        return redirect('cart')
    return redirect('cart')

@login_required
def add_to_account(request):
    user = request.user
    order_qs = Order.objects.filter(user=user, ordered=False)
    if order_qs.exists():
        order = order_qs[0]
        order_of_books = order.books.filter(user=user)
        for book_order in order_of_books:
            user.profile.books.add(book_order.book)
        messages.success(request, f'Books has benn added to your profile')
        order.delete()
    else:
        messages.warning(request, f"Order dosen't exists")
        return redirect('cart')
    return redirect('cart')

class CommentCreateView(LoginRequiredMixin, CreateView):
    model = BookComments
    template_name = 'library/create_comment.html'
    fields = ['book','content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        form.instance.post_date = timezone.now()
        messages.success(self.request, f'You added new comment')
        return super().form_valid(form)

@login_required
def book_detail_comment_view(request, slug):
    book = get_object_or_404(Book, slug=slug)
    new_comment = BookComments

    if request.method == 'POST':
        comment_form = CommentCreateForm(request.POST, instance=request.user)
        if comment_form.is_valid():
            comment_form.save()
            messages.success(request, f'You added new comment!')
            return redirect('library-detail', slug=slug)
    else: 
        comment_form = CommentCreateForm()

    context = {
        'book': book,
        'comment_form': comment_form
    }
    return render(request, 'library/detail.html', context)