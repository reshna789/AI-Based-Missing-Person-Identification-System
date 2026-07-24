from django import forms
from .models import VideoAnalysis
from cases.models import MissingPersonCase


class VideoUploadForm(forms.ModelForm):
    """Form for police to upload CCTV footage for analysis."""
    case = forms.ModelChoiceField(
        queryset=MissingPersonCase.objects.filter(status__in=['active', 'investigating']),
        required=False,
        empty_label="-- Scan against ALL active cases --",
        help_text="Select a specific case to search for, or leave blank to scan all."
    )

    class Meta:
        model = VideoAnalysis
        fields = ['video', 'case']
        widgets = {
            'video': forms.ClearableFileInput(attrs={'accept': 'video/*'}),
        }
