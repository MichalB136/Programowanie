from django import template
from library.models import Order

register = template.Library()

@register.filter(name='item_count')
def cart_item_count(user):
    if user.is_authenticated:
        qs = Order.objects.filter(user=user, ordered=False)
        if qs.exists():
            return qs[0].books.count()
    return 0


@register.filter
def moja_funkcja_liczaca_cene(user):
    if user.is_authenticated:
        orders = Order.objects.filter(user=user, ordered=False)
        price = 0
        if orders.exists():
            book_orders = orders[0].books.all()
            for book_order in book_orders:
                price += book_order.book.price
            return price
        return price 