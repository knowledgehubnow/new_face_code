from rest_framework import serializers
from .models import *

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = ('user','video_file',)

class VideoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"
    
class VideoDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = ('image',)

class ImageDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = "__all__"

class ImageDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = "__all__"

class TotalFrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Frame
        fields = "__all__"
        
class VideoFrameSerializer(serializers.ModelSerializer):
    frames = TotalFrameSerializer(many=True)
    class Meta:
        model = DetectedFrames
        fields = "__all__"




