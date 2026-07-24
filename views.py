import threading
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from cases.views import role_required
from .models import VideoAnalysis
from .forms import VideoUploadForm
from .services import process_video_analysis


# Global thread pool for video analysis
# Allows up to 4 videos to be processed simultaneously for faster multi-video throughput
video_analysis_executor = ThreadPoolExecutor(max_workers=4)


@role_required('police')
def upload_video(request):
    
    from cases.models import MissingPersonCase
    if request.method == 'POST':
        videos = request.FILES.getlist('videos')
        case_id = request.POST.get('case')

        if not case_id:
            messages.error(request, 'Please select a registered case.')
        elif not videos:
            messages.error(request, 'Please select at least one video file.')
        else:
            case = MissingPersonCase.objects.filter(id=case_id).first()
            if not case:
                messages.error(request, 'Selected case not found.')
            else:
                is_cctv = 'cctv_analysis' in request.POST
                for video_file in videos:
                    analysis = VideoAnalysis(
                        uploaded_by=request.user,
                        video=video_file,
                        case=case,
                        is_cctv=is_cctv
                    )
                    analysis.status = 'processing'
                    analysis.save()

                    # Submit analysis task to background thread pool
                    video_analysis_executor.submit(process_video_analysis, analysis)

                messages.success(request, f'{len(videos)} video(s) uploaded for case "{case.full_name}". Analysis is running in the background.')
                return redirect('analysis_list')

    cases = MissingPersonCase.objects.filter(status__in=['active', 'investigating'])
    return render(request, 'ai_engine/upload_video.html', {'cases': cases})


@role_required('police')
def analysis_detail(request, analysis_id):
   
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    matches = analysis.matches.all()
    return render(request, 'ai_engine/analysis_detail.html', {
        'analysis': analysis,
        'matches': matches,
    })


@role_required('police')
def rerun_analysis(request, analysis_id):
    if request.method == 'POST':
        analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
        
        # Reset analysis state
        analysis.matches.all().delete()
        analysis.status = 'processing'
        analysis.progress = 0
        analysis.completed_at = None
        analysis.error_message = None
        analysis.save()
        
        # Resubmit to background thread
        video_analysis_executor.submit(process_video_analysis, analysis)
        
        messages.success(request, 'Analysis has been restarted.')
        return redirect('analysis_detail', analysis_id=analysis.id)
    return redirect('analysis_detail', analysis_id=analysis_id)



@role_required('police')
def analysis_list(request):
    analyses = VideoAnalysis.objects.all().order_by('-created_at')
    
    # Filtering
    q = request.GET.get('q')
    status = request.GET.get('status')
    result = request.GET.get('result')
    date = request.GET.get('date')

    if q:
        analyses = analyses.filter(case__full_name__icontains=q)
    if status:
        analyses = analyses.filter(status=status)
    if result:
        if result == 'found':
            analyses = analyses.filter(matches__isnull=False).distinct()
        elif result == 'not_found':
            analyses = analyses.filter(status='completed', matches__isnull=True)
        elif result == 'processing':
            analyses = analyses.filter(status='processing')
    if date:
        analyses = analyses.filter(created_at__date=date)

    # Summary Stats
    all_analyses = VideoAnalysis.objects.all()
    stats = {
        'total': all_analyses.count(),
        'processing': all_analyses.filter(status='processing').count(),
        'match_found': all_analyses.filter(matches__isnull=False).distinct().count(),
        'match_not_found': all_analyses.filter(status='completed', matches__isnull=True).count(),
        'failed': all_analyses.filter(status='failed').count(),
    }

    return render(request, 'ai_engine/analysis_list.html', {
        'analyses': analyses,
        'stats': stats,
        # Pass filters back to template
        'q': q,
        'status_filter': status,
        'result_filter': result,
        'date_filter': date,
    })


@role_required('police')
def analysis_status_api(request, analysis_id):
    
    from django.http import JsonResponse
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    return JsonResponse({
        'status': analysis.status,
        'progress': analysis.progress,
        'match_count': analysis.matches.count(),
    })
