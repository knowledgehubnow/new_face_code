from django.contrib import admin
from django.urls import path,include
from . import views
from .views import *

urlpatterns = [
    path('',views.scan_face,name = "scan_face"),
    path('image/processing/',views.image_analysis,name = "image_analysis"),
    path('analyzed/video/view/<int:video_id>/',views.analized_video_detail,name = "analized_video_detail"),
    path('analized/video/list/',views.analized_video_list,name = "analized_video_list"),
    path('analized/image/list/',views.analized_image_list,name = "analized_image_list"),
    path('analyze/pdf/',views.analyze_pdf,name = "analyze_pdf"),
    path('analyzed/pdf/list/',views.analyzed_pdf_list,name = "analyzed_pdf_list"),
    path('analyzed/pdf/view/<int:pdf_id>/',views.analyzed_pdf_view,name = "analyzed_pdf_view"),
    
    path('upload/', VideoUploadView.as_view()),
    path('image/analysis/', ImageAnalysisView.as_view()),
    path('analysed/video/list/', AnalysedVideoListView.as_view()),
    path('analysed/video/detail/view/', AnalysedVideoDetailView.as_view()),
    path('analysed/image/list/', AnalysedImageListView.as_view()),
    path('video/frame/view/', VideoFrameView.as_view()),

    path('video_detail/<int:video_id>/',views.video_detail,name = "video_detail"),
    path('delete/data/',views.delete_data,name = "analyzed_pdf_view"),
    path('frame/data/',views.frame,name = "frame"),
    path('get/data/<int:id>/',views.get_data,name = "get_data"),
]
