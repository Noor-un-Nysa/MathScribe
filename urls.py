
from django.urls import path
from . import views

urlpatterns = [
    path('', views.draw_view, name='draw'),
    path('predict/', views.predict_view, name='predict'),
]

