from django.db import models

# Create your models here.

class Author(models.Model):
    a = models.CharField(("a"), max_length=50)

class Book(models.Model):
    a = models.CharField(("a"), max_length=50)

class Genre(models.Model):
    a = models.CharField(("a"), max_length=50)

class Order(models.Model):
    a = models.CharField(("a"), max_length=50)

class OrderBook(models.Model):
    a = models.CharField(("a"), max_length=50)
