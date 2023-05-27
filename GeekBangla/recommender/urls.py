from django.urls import path, re_path
from recommender import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns=[
    re_path(r'^recommend', views.recommend),
    re_path(r'^train', views.train),
    re_path(r'^test', views.test),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 