"""
AI Face Recognition Service — V9 DeepFace Engine  ⚡

Replaced face_recognition (dlib) with DeepFace so the application runs
with the global Python installation where dlib cannot be compiled.

Key changes over V8:
  1. Uses DeepFace.represent() for 512-d Facenet embeddings (cosine distance)
  2. Uses RetinaFace detector (falls back to MTCNN, then OpenCV)
  3. No dlib / face_recognition dependency
  4. Sequential frame-grab optimisation retained from V8
  5. Reference encoding cache retained from V8
  6. Batch vectorised cosine comparison retained

Expected: 60s video → <3s processing (TF model warm-up on first run)
"""

import os
import time
import cv2
import numpy as np
import threading
from collections import deque
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile
from django.utils import timezone
from django import db

# DeepFace is imported lazily (inside functions) to avoid blocking Django
# startup with TensorFlow initialisation time.


# ========================== Configuration ==========================
NORMAL_FRAME_SKIP = 10        # Skip 10 frames (process ~3 fps)
CCTV_FRAME_SKIP   = 6         # Skip 6 frames (process ~4 fps) to optimize speed
RESIZE_SCALE      = 0.6       # Balanced resize for speed/accuracy on normal video
NORMAL_THRESHOLD  = 0.35      # Cosine distance (lower is more similar)
CCTV_THRESHOLD    = 0.38      # Tuned precisely to prevent False Positives without missing True Positives
MIN_FACE_PX       = 15        # Catch smaller faces in distant CCTV
PREFETCH_BUFFER   = 32        # Larger buffer for smoother threading
EMBEDDING_MODEL   = "Facenet512" 
# Detectors
NORMAL_DETECTOR   = "mtcnn"       # Fast and accurate
CCTV_DETECTOR     = "mtcnn"       # Ultra-fast and accurate for CCTV distances. Retinaface was too slow.
# ===================================================================


# ============ Shared one-time resources ============
_ssd_net_lock  = threading.Lock()
_ssd_net       = None            # OpenCV DNN SSD (loaded once, fallback only)

_encoding_cache      = {}        # {case_id → [512-d numpy arrays]}
_encoding_cache_lock = threading.Lock()

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_GAMMA_LUT = np.array([((i / 255.0) ** (1.0 / 1.4)) * 255
                        for i in range(256)]).astype("uint8")

# ===================================================

def _get_deepface():
    """Lazy import of DeepFace — only pays the TF init cost on first call."""
    from deepface import DeepFace as _DF  # noqa: PLC0415
    return _DF


# --------------- timing helper ---------------

class _Timer:
    def __init__(self, label):
        self._label = label
        self._steps = []
        self._t = time.perf_counter()

    def mark(self, step_name):
        now = time.perf_counter()
        elapsed = (now - self._t) * 1000
        self._steps.append((step_name, elapsed))
        self._t = now
        return elapsed

    def summary(self):
        total = sum(ms for _, ms in self._steps)
        parts = " | ".join(f"{name}={ms:.1f}ms" for name, ms in self._steps)
        print(f"[TIMER] {self._label}: {parts} | TOTAL={total:.1f}ms")


# --------------- SSD fallback (OpenCV DNN) ---------------

def _load_ssd():
    global _ssd_net
    with _ssd_net_lock:
        if _ssd_net is not None:
            return _ssd_net
        proto   = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
        weights = os.path.join(os.path.dirname(__file__),
                               'res10_300x300_ssd_iter_140000.caffemodel')
        if os.path.exists(proto) and os.path.exists(weights):
            _ssd_net = cv2.dnn.readNetFromCaffe(proto, weights)
            print("[AI ENGINE] DNN SSD loaded.")
        else:
            _ssd_net = "fallback"
            print("[AI ENGINE] SSD files missing — using HOG fallback.")
    return _ssd_net


def _detect_ssd_opencv(frame_bgr, conf_thresh=0.40):
    """OpenCV DNN SSD detection. Returns [(x,y,w,h), ...]."""
    net = _load_ssd()
    if net == "fallback":
        return []
    fh, fw = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        c = float(dets[0, 0, i, 2])
        if c < conf_thresh:
            continue
        b = (dets[0, 0, i, 3:7] * np.array([fw, fh, fw, fh])).astype(int)
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
            boxes.append((max(0, x1), max(0, y1), w, h))
    return boxes


# --------------- DeepFace detection & encoding ---------------

def _detect_and_encode_deepface(frame_bgr, detector="retinaface"):
    """
    Detect faces in frame_bgr using DeepFace and return their 512-d embeddings
    along with bounding boxes.

    Returns list of (encoding_array, (x, y, w, h)) tuples.
    """
    try:
        DeepFace = _get_deepface()
        results = DeepFace.represent(
            img_path=frame_bgr,        # accepts numpy BGR array
            model_name=EMBEDDING_MODEL,
            detector_backend=detector,
            enforce_detection=False,   # don't raise if no face found
            align=True,
        )
        out = []
        for r in results:
            if not r.get("embedding"):
                continue
            enc = np.array(r["embedding"], dtype=np.float32)
            fa  = r.get("facial_area", {})
            x   = fa.get("x", 0)
            y   = fa.get("y", 0)
            w   = fa.get("w", 0)
            h   = fa.get("h", 0)
            if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
                # Reject fallback 'whole image' embeddings from enforce_detection=False
                # If width and height are nearly the whole original frame buffer, it's not a face
                img_h, img_w = frame_bgr.shape[:2]
                if w > img_w * 0.9 and h > img_h * 0.9:
                    continue
                out.append((enc, (x, y, w, h)))
        return out
    except Exception as e:
        print(f"[AI ENGINE] detect_encode error ({detector}): {e}")
        return []


def _generate_reference_encodings(img_bgr):
    """
    Compute DeepFace Facenet512 embeddings for a reference (case photo) image.
    Also includes a horizontally flipped version for profile coverage.
    Returns list of 512-d numpy arrays.
    """
    DeepFace = _get_deepface()
    _ref_backends = ["retinaface", "mtcnn", "opencv"]
    encodings = []
    for frame in [img_bgr, cv2.flip(img_bgr, 1)]:
        for backend in _ref_backends:
            try:
                results = DeepFace.represent(
                    img_path=frame,
                    model_name=EMBEDDING_MODEL,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                )
                for r in results:
                    if r.get("embedding"):
                        enc = np.array(r["embedding"], dtype=np.float32)
                        encodings.append(enc)
                if encodings:
                    break   # got at least one — no need to try next backend
            except Exception:
                continue
    return encodings


def _cosine_distances(cand_matrix, ref_matrix):
    """
    Vectorised cosine distance between every candidate and every reference.
    cand_matrix : (C, D)
    ref_matrix  : (N, D)
    Returns (C, N) distance matrix (0 = identical, 1 = totally different).
    """
    # Normalise rows
    cn = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-9)
    rn = ref_matrix  / (np.linalg.norm(ref_matrix,  axis=1, keepdims=True) + 1e-9)
    # cosine distance = 1 - cosine similarity
    return 1.0 - cn @ rn.T   # (C, N)


# --------------- CCTV preprocessing ---------------

def _preprocess_cctv_enhanced(frame_bgr):
    """
    Enhanced CCTV preprocessing: Mild Contrast → Gamma.
    Avoids aggressive sharpening which corrupts Facenet embeddings.
    """
    # 1. Local Contrast (CLAHE) - very gentle to avoid noise amplification
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Dynamic Gamma (Auto-correction for dark scenes)
    img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(img_gray)
    if mean_brightness < 100:  # If dark, apply stronger gamma
        gamma = 1.6
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in range(256)]).astype("uint8")
        frame_bgr = cv2.LUT(frame_bgr, table)
    
    return frame_bgr


# --------------- Frame prefetch buffer (sequential grab) ---------------

class _FrameBuffer:
    """
    Background-thread frame reader using sequential grab/retrieve.
    Avoids costly keyframe seeking on H.264/H.265 video.
    """
    def __init__(self, cap, frame_skip, maxlen=PREFETCH_BUFFER):
        self._cap   = cap
        self._skip  = frame_skip
        self._buf   = deque(maxlen=maxlen)
        self._done  = False
        self._lock  = threading.Lock()
        self._cond  = threading.Condition(self._lock)
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._t     = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self):
        frame_num = 0
        while frame_num < self._total:
            if self._done:
                return
            if frame_num % self._skip == 0:
                ok, raw = self._cap.read()
                if not ok:
                    break
                ts = frame_num / self._fps
                with self._cond:
                    while len(self._buf) >= self._buf.maxlen and not self._done:
                        self._cond.wait(timeout=0.1)
                    if self._done:
                        return
                    self._buf.append((frame_num, ts, raw))
                    self._cond.notify_all()
            else:
                if not self._cap.grab():
                    break
            frame_num += 1
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def stop(self):
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def __iter__(self):
        while True:
            with self._cond:
                while not self._buf and not self._done:
                    self._cond.wait(timeout=0.1)
                if self._buf:
                    yield self._buf.popleft()
                    self._cond.notify_all()
                elif self._done:
                    return


# --------------- main entry point ---------------

def process_video_analysis(analysis):
    """
    Main analysis function — called from background ThreadPoolExecutor in views.py.

    V9 changes (DeepFace engine):
      - Replaced face_recognition/dlib with DeepFace (Facenet512 + RetinaFace)
      - Cosine distance instead of Euclidean for matching
      - No dlib / cmake dependency — works with global Python
      - All V8 perf optimisations retained
    """
    from .models import MatchResult, VideoAnalysis

    try:
        t_total = time.perf_counter()
        db.close_old_connections()

        # Verify DeepFace is importable before doing any DB work
        try:
            _get_deepface()
        except ImportError:
            raise RuntimeError(
                "DeepFace is not installed. Run: pip install deepface"
            )

        analysis = VideoAnalysis.objects.select_related('case').get(id=analysis.id)
        analysis.status = 'processing'
        analysis.save(update_fields=['status'])

        video_path = analysis.video.path
        case       = analysis.case
        is_cctv    = getattr(analysis, 'is_cctv', False)
        threshold  = CCTV_THRESHOLD if is_cctv else NORMAL_THRESHOLD

        # ── STEP 1: Reference Encodings (cached per case) ──────────────
        t_ref     = time.perf_counter()
        cache_key = case.id
        with _encoding_cache_lock:
            cached = _encoding_cache.get(cache_key)

        if cached is not None:
            ref_encodings = cached
            print(f"[AI ENGINE] >>> Cached encodings for case {cache_key} "
                  f"({len(ref_encodings)} enc)")
        else:
            ref_encodings = []
            for img_obj in case.images.all():
                img_bgr = cv2.imread(img_obj.image.path)
                if img_bgr is None:
                    continue
                try:
                    encs = _generate_reference_encodings(img_bgr)
                    ref_encodings.extend(encs)
                    print(f"[AI ENGINE] [+] Ref encoded: "
                          f"{os.path.basename(img_obj.image.path)} "
                          f"({len(encs)} face(s))")
                except Exception as e:
                    print(f"[AI ENGINE] ref encode error: {e}")
                    continue
            with _encoding_cache_lock:
                _encoding_cache[cache_key] = ref_encodings

        ref_ms = (time.perf_counter() - t_ref) * 1000
        print(f"[AI ENGINE] Reference encodings: {len(ref_encodings)} "
              f"enc in {ref_ms:.0f}ms")

        if not ref_encodings:
            analysis.status        = 'completed'
            analysis.error_message = "No faces found in case photos."
            analysis.save(update_fields=['status', 'error_message'])
            return

        ref_matrix = np.array(ref_encodings, dtype=np.float32)   # (N, 512)

        mode = 'CCTV' if is_cctv else 'Normal'
        frame_skip_val = CCTV_FRAME_SKIP if is_cctv else NORMAL_FRAME_SKIP
        detector = CCTV_DETECTOR if is_cctv else NORMAL_DETECTOR
        print(f"[AI ENGINE] Starting {mode} analysis | "
              f"threshold={threshold} | skip={frame_skip_val} | det={detector}")

        # ── STEP 2: Video Scan ──────────────────────────────────────────
        t_video = time.perf_counter()
        cap     = cv2.VideoCapture(video_path)
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf     = _FrameBuffer(cap, frame_skip=frame_skip_val)

        open_ms = (time.perf_counter() - t_video) * 1000
        print(f"[AI ENGINE] Video opened: {total} frames, {fps:.0f} fps, "
              f"{total/fps:.1f}s duration ({open_ms:.0f}ms)")

        frames_done   = 0
        faces_checked = 0

        # ── Verification Counters (Reduce False Positives) ────────────────
        match_hits = 0
        best_match_so_far = None

        for frame_num, ts, raw_bgr in buf:
            # Update progress periodically
            current_progress = int((frame_num / total) * 100) if total > 0 else 0
            if current_progress != analysis.progress:
                analysis.progress = current_progress
                analysis.save(update_fields=['progress'])

            timer = _Timer(f"frame#{frame_num}")

            # ── Preprocess ───────────────────────────────────────────
            if is_cctv:
                proc_bgr = _preprocess_cctv_enhanced(raw_bgr)
            else:
                # Basic enhancement for normal video too
                lab = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _CLAHE.apply(l)
                proc_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            timer.mark("preproc")

            # ── Resize ───────────────────────────────────────────────
            # Use strict 1.0 scale for CCTV to prevent decimating distant/small faces
            h, w = proc_bgr.shape[:2]
            current_scale = 1.0 if is_cctv else RESIZE_SCALE
            
            if current_scale != 1.0:
                small_bgr = cv2.resize(proc_bgr, (0, 0),
                                       fx=current_scale, fy=current_scale,
                                       interpolation=cv2.INTER_LINEAR)
            else:
                small_bgr = proc_bgr
            timer.mark("resize")

            # ── Detect + Encode (DeepFace) ───────────────────────────
            face_pairs = _detect_and_encode_deepface(small_bgr, detector)

            # Fallback for reliability if primary detector fails on a frame
            if not face_pairs and detector != "opencv":
                face_pairs = _detect_and_encode_deepface(small_bgr, "opencv")

            timer.mark("detect+encode")

            frames_done += 1
            if not face_pairs:
                if frames_done % 30 == 0:
                    timer.summary()
                continue

            # ── Compare ──────────────────────────────────────────────
            candidate_encodings = [enc for enc, _ in face_pairs]
            candidate_boxes     = [box for _, box in face_pairs]

            cand_matrix = np.array(candidate_encodings, dtype=np.float32)
            dist_matrix = _cosine_distances(cand_matrix, ref_matrix)
            min_dists   = dist_matrix.min(axis=1)
            best_idx    = int(np.argmin(min_dists))
            min_dist    = float(min_dists[best_idx])
            timer.mark("compare")

            faces_checked += len(candidate_encodings)

            if min_dist < threshold:
                confidence = round(max(0.0, (1.0 - min_dist)) * 100, 2)
                print(f"[AI ENGINE] MATCH CANDIDATE at {ts:.2f}s | dist={min_dist:.4f} | conf={confidence}%")
                
                # Scaled coordinates
                sx, sy, sw, sh = candidate_boxes[best_idx]
                inv = 1.0 / current_scale
                x, y = int(sx * inv), int(sy * inv)
                w, h = int(sw * inv), int(sh * inv)

                # Store result
                match_hits += 1
                if best_match_so_far is None or min_dist < best_match_so_far['dist']:
                    best_match_so_far = {
                        'ts': ts,
                        'conf': confidence,
                        'dist': min_dist,
                        'box': (x, y, w, h),
                        'raw': raw_bgr
                    }

                # Speed optimization: if we have a very strong match, exit early
                if min_dist < (threshold * 0.85):
                    print(f"[AI ENGINE] Strong match found! Exiting early.")
                    break
                
                # Otherwise keep looking for a bit more to be sure if noise is high
                if match_hits >= 2:
                    print(f"[AI ENGINE] Confirmed match (2 hits). Exiting.")
                    break

        cap.release()

        total_ms = (time.perf_counter() - t_total) * 1000
        print(f"[AI ENGINE] ========================================")
        print(f"[AI ENGINE] DONE in {total_ms:.0f}ms | Frames: {frames_done} | Faces: {faces_checked} | Hits: {match_hits}")
        print(f"[AI ENGINE] ========================================")

        if best_match_so_far:
            ts = best_match_so_far['ts']
            confidence = best_match_so_far['conf']
            x, y, w, h = best_match_so_far['box']
            annotated = best_match_so_far['raw'].copy()
            
            # Annotation
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(annotated, f"IDENTIFIED: {confidence}%", (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            buf_io = BytesIO()
            Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(buf_io, format='JPEG', quality=85)

            match = MatchResult(
                analysis=analysis,
                matched_case=case,
                confidence=confidence,
                timestamp_seconds=round(ts, 2),
            )
            match.frame_image.save(f'match_{analysis.id}.jpg', ContentFile(buf_io.getvalue()), save=False)
            match.save()

            analysis.status = 'completed'
            analysis.progress = 100
            analysis.completed_at = timezone.now()
            analysis.error_message = f"Match Found! Person identified at {ts:.1f}s with {confidence}% similarity."
        else:
            analysis.status = 'completed'
            analysis.progress = 100
            analysis.completed_at = timezone.now()
            analysis.error_message = "Match Not Found. The person was not identified in this footage."
        
        analysis.save()

    except Exception as e:
        import traceback
        analysis.status        = 'failed'
        analysis.progress      = 100
        analysis.error_message = str(e)
        analysis.save()
        print(f"[AI ENGINE] ERROR: {e}")
        traceback.print_exc()
    finally:
        db.close_old_connections()
