# Generated by Django 3.0.8 on 2020-08-11 17:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0008_auto_20200811_1807'),
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='books',
            field=models.ManyToManyField(to='library.Book'),
        ),
    ]