from django.contrib import admin
from .models import VideoAnalysis, MatchResult


class MatchResultInline(admin.TabularInline):
    model = MatchResult
    extra = 0
    readonly_fields = ('matched_case', 'confidence', 'timestamp_seconds', 'frame_image')


@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'uploaded_by', 'case', 'status', 'created_at', 'completed_at')
    list_filter = ('status', 'created_at')
    inlines = [MatchResultInline]


@admin.register(MatchResult)
class MatchResultAdmin(admin.ModelAdmin):
    list_display = ('analysis', 'matched_case', 'confidence', 'timestamp_seconds')
    list_filter = ('matched_case',)
