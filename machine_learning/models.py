from django.db import models
from django.conf import settings
import ast
import json
from django.utils import timezone
# Create your models here.


class ImageRecognition(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    image = models.ImageField(upload_to='images/', blank=True, null=True)
    image_analysis_data = models.JSONField(null=True, blank=True)
    dominant_emotion = models.CharField(max_length=255, null=True, blank=True)

    def images_list(self):
        return ast.literal_eval(self.dominant_emotion)

class VideoRecognition(models.Model):
    created_date = models.DateTimeField(default=timezone.localtime)
    user = models.BigIntegerField(null = True)
    name = models.CharField(max_length=255, null=True, blank=True)
    thumb_img = models.ImageField(upload_to='thumbnails/', null=True, blank=True)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    video_durations = models.FloatField(null=True, blank=True)
    analysis_score = models.FloatField(default=0.0)
    word_per_minute = models.FloatField(null=True, blank=True)
    language_analysis = models.CharField(max_length = 100,null=True, blank=True)
    voice_modulation_analysis = models.JSONField(null=True, blank=True, default=None)
    energy_level_analysis = models.CharField(max_length=255, null=True, blank=True)
    filler_words_used = models.JSONField(null=True, blank=True)
    frequently_used_word = models.JSONField(null=True, blank=True)
    voice_emotion = models.JSONField(null=True, blank=True, default=None)
    confidence = models.CharField(max_length=100,null=True, blank=True)
    facial_expression = models.CharField(max_length=100,null=True, blank=True)
    eye_bling = models.CharField(max_length=100,null=True, blank=True)
    hand_movement = models.CharField(max_length=100,null=True, blank=True)
    eye_contact = models.CharField(max_length=100,null=True, blank=True)
    thanks_gesture = models.CharField(max_length=100,null=True, blank=True)
    greeting = models.CharField(max_length=100,null=True, blank=True)
    greeting_gesture = models.CharField(max_length=100,null=True, blank=True)
    voice_tone = models.CharField(max_length=100,null=True, blank=True)
    voice_pauses = models.CharField(max_length=100,null=True, blank=True)
    appropriate_facial = models.CharField(max_length=100,null=True, blank=True)
    body_posture = models.CharField(max_length=100,null=True, blank=True)
    body_language_score = models.FloatField(null=True, blank=True)
    facial_expression_score = models.FloatField(null=True, blank=True)
    language_analysis_score = models.FloatField(null=True, blank=True)
    voice_modulation_score = models.FloatField(null=True, blank=True)
    body_confidence_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name
    
    
class Posture(models.Model):
    video = models.ForeignKey(VideoRecognition, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name

class DetectedFrames(models.Model):
    posture = models.ForeignKey(Posture, on_delete=models.CASCADE)
    frames = models.ManyToManyField('Frame', blank=True)

    def __str__(self):
        return self.posture.name

class Frame(models.Model):
    image = models.ImageField(upload_to='frames/', null=True, blank=True)
    number = models.IntegerField(null=True, blank=True)
    current_time = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Frame {self.pk}" 


class AnalyzePDF(models.Model):
    pdf_name = models.CharField(max_length=255, null=True, blank=True)
    pdf_file = models.FileField(upload_to='pdf/', null=True, blank=True)
    pdf_text = models.TextField()

    def __str__(self):
        return self.pdf_name
    

    
