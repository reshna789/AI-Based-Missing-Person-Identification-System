from django.urls import path
from . import views

urlpatterns = [
    path('police/upload-video/', views.upload_video, name='upload_video'),
    path('police/analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('police/analysis/<int:analysis_id>/rerun/', views.rerun_analysis, name='rerun_analysis'),
    path('police/analyses/', views.analysis_list, name='analysis_list'),
    path('api/analysis/<int:analysis_id>/status/', views.analysis_status_api, name='analysis_status_api'),
]
