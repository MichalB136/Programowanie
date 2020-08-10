from django.db import models
from django.urls import reverse
from django.utils.text import slugify
from PIL import Image

# Create your models here.

class Author(models.Model):
    first_name = models.CharField(max_length=25)
    last_name = models.CharField(max_length=25)
    
    def __str__(self):
        return f'{self.first_name} {self.last_name}'

class Genre(models.Model):
    genre = models.CharField(max_length=20)

    def __str__(self):
        return self.genre

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    genre = models.ManyToManyField(Genre)
    description = models.TextField(default="This description hasn't been added yet.")
    image = models.ImageField(default='book_default.jpg', upload_to='book_pics')
    slug = models.SlugField(default=' ', max_length=100)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("library-detail", kwargs={"slug": self.slug})
    
    def save(self, *args, **kwargs):
        value = self.title
        self.slug = slugify(value, allow_unicode=True)
        img = Image.open(self.image.path)
        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)
        super().save( *args, **kwargs)



class Order(models.Model):
    a = models.CharField(("a"), max_length=50)

class OrderBook(models.Model):
    a = models.CharField(("a"), max_length=50)
