from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('users.urls')),
    path('', include('cases.urls')),
    path('', include('ai_engine.urls')),
]

# Force serving static and media files locally even when DEBUG=False 
# (Useful for performance testing without breaking CSS/Images)
from django.urls import re_path
from django.views.static import serve

urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
]
