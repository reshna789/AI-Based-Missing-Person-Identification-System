from django.db import models
from django.conf import settings
from cases.models import MissingPersonCase


class VideoAnalysis(models.Model):
    """A video uploaded by police for AI-based face recognition analysis."""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='video_analyses'
    )
    video = models.FileField(upload_to='analysis_videos/')
    case = models.ForeignKey(
        MissingPersonCase,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='video_analyses',
        help_text="Optionally target a specific case, or leave blank to scan against all active cases."
    )
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0, help_text="Analysis progress percentage (0-100)")
    is_cctv = models.BooleanField(default=False, help_text="True if this analysis should apply CCTV-specific AI methods")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Video Analyses"

    def __str__(self):
        target = self.case.full_name if self.case else "All Cases"
        return f"Video Analysis #{self.id} - {target}"


class MatchResult(models.Model):
    """A face match detected by the AI engine."""
    analysis = models.ForeignKey(VideoAnalysis, on_delete=models.CASCADE, related_name='matches')
    matched_case = models.ForeignKey(MissingPersonCase, on_delete=models.CASCADE, related_name='match_results')
    confidence = models.FloatField(help_text="Match confidence score (0-100%)")
    timestamp_seconds = models.FloatField(help_text="Timestamp in video where the face was detected (seconds)")
    frame_image = models.ImageField(upload_to='match_frames/', help_text="Snapshot of the matched frame")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-confidence']

    def __str__(self):
        return f"Match: {self.matched_case.full_name} ({self.confidence:.1f}%) at {self.timestamp_seconds:.1f}s"
