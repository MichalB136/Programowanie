# Generated by Django 3.0.8 on 2020-08-20 11:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0013_auto_20200820_1322'),
    ]

    operations = [
        migrations.RenameField(
            model_name='bookcomments',
            old_name='user',
            new_name='author',
        ),
    ]