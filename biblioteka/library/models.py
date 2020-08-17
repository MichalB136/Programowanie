from django.db import models
from django.urls import reverse
from django.utils.text import slugify
from django.contrib.auth.models import User
from PIL import Image

# Create your models here.

class Author(models.Model):
    first_name = models.CharField(max_length=25)
    last_name = models.CharField(max_length=25)
    
    def __str__(self):
        return f'{self.first_name} {self.last_name}'

    def get_absolute_url(self):
        return reverse("library-create")

class Genre(models.Model):
    genre = models.CharField(max_length=20, unique=True)

    def __str__(self):
        return self.genre

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    genre = models.ManyToManyField(Genre)
    description = models.TextField(default="This description hasn't been added yet.")
    image = models.ImageField(default='book_default.jpg', upload_to='book_pics')
    price = models.FloatField(default=0)
    slug = models.SlugField(default=' ', max_length=100)
    
    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("library-detail", kwargs={"slug": self.slug})

    def get_add_to_cart_url(self):
        return reverse("add-to-cart", kwargs={"slug": self.slug})
    
    def get_remove_from_cart(self):
        return reverse("remove-from-cart", kwargs={"slug": self.slug})
    
    def save(self, *args, **kwargs):
        value = self.title
        self.slug = slugify(value, allow_unicode=True)
        img = Image.open(self.image.path)
        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)
        super().save( *args, **kwargs)



class OrderBook(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    ordered = models.BooleanField(default=False)

    def __str__(self):
        return f'Order of {self.book.title}'
    
class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    books = models.ManyToManyField(OrderBook)
    start_date = models.DateTimeField(auto_now_add=True)
    ordered_date = models.DateTimeField()
    ordered = models.BooleanField(default=False)

    def __str__(self):
        return f'Orders for {self.user.username}'

    def get_add_to_account(self):
        return reverse("add-to-account")
    
