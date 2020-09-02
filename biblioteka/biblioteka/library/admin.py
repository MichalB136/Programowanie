from django.contrib import admin
from .models import Book, Author, Genre, Order, OrderBook, BookComments

class BookAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("title",)}

admin.site.register(Book, BookAdmin)
admin.site.register(Author)
admin.site.register(Genre)
admin.site.register(Order)
admin.site.register(OrderBook)
admin.site.register(BookComments)
