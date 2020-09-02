"""
WSGI config for biblioteka project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os, sys

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'biblioteka.settings')
sys.path.append('/home/pi/librarys')
sys.path.append('/home/pi/librarys/biblioteka')

application = get_wsgi_application()
