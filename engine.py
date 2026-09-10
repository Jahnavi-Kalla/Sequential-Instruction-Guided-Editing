#engine
import os, math, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import torch
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
SEED     = 1234
STEPS    = 40
CFG      = 4.0

NEG_PRESERVE = ("additional people, extra person, extra hands, extra fingers, "
                "deformed hands, deformed face, extra clutter, duplicated objects, "
                "deformed or warped objects, floating objects, text, watermark, "
                "changed background, changed room, changed clothing, blurry, "
                "distorted perspective")

NEG_STRICT_NO_PEOPLE = ("a person, people, hands, extra clutter, duplicated objects, "
                        "deformed or warped objects, floating objects, text, watermark, "
                        "changed background, changed room, blurry, distorted perspective")

NEG_UNIVERSAL = (
    "deformed, warped, distorted, melted, smeared, blurry, out of focus, "
    "low quality, jpeg artifacts, noisy, grainy, oversaturated, "
    "extra fingers, deformed hands, duplicated object, cloned object, "
    "floating object, disconnected parts, wrong perspective, "
    "inconsistent lighting, changed background, changed room, "
    "watermark, text, signature, frame, border"
)

NEG = NEG_UNIVERSAL

NEG_WITH_PERSON = NEG_PRESERVE
NEG_NO_PERSON   = NEG_STRICT_NO_PEOPLE

IN_DIR   = "generated_input"
MANIFEST = os.path.join(IN_DIR, "manifest.txt")
PROJECT_MANIFOLD = False
NULL_INSTRUCTION = "Return the exact same image with no changes at all."
NULL_STEPS       = 40
NULL_CFG         = 1.0

ENHANCE_DISPLAY  = True
ENHANCE_SCALE    = 2
ESRGAN_WEIGHTS   = "weights/RealESRGAN_x4plus.pth"
SHEET_CELL_DISPLAY = 900

OUT_ROOT  = "full_2511_gen1"
TASK_NAME = "tray_sam"

TASK_STEPS = [
    "Place one bright red plastic tray flat on the empty wooden workbench "
    "surface in the foreground, in the middle of the clear area, keeping the "
    "room unchanged.",

    "Place one large bright blue closed plastic box flat on the workbench to "
    "the right of the same red tray, close to it but not touching it, keeping "
    "the room unchanged.",

    "Place one bright yellow rectangular wooden block on the workbench just in "
    "front of the same red tray, closer to the camera, not overlapping the "
    "tray, keeping the room unchanged.",

    "Place one bright green coffee mug on the workbench to the left of the same "
    "red tray, standing upright, not touching the tray, keeping the room "
    "unchanged.",

    "Place one large orange rectangular sponge flat on top of the same blue "
    "box, fully visible and centred on the box lid, keeping the room unchanged.",

    "Place one white folded cloth flat on the workbench in front of the same "
    "blue box, closer to the camera, not overlapping any other object, keeping "
    "the room unchanged.",
]

VARIANT      = TASK_NAME
VARIANTS     = {TASK_NAME: TASK_STEPS}
INSTRUCTIONS = VARIANTS[VARIANT]

INSTRUCTIONS_TRAY_SAM = TASK_STEPS
import time
from contextlib import contextmanager

TIMING_ENABLED = True
_TIMING = []


def _sync():
    if TIMING_ENABLED and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(section, **meta):
    """with timed('edit', stem=stem, step=i): ...   -> appends to _TIMING."""
    if not TIMING_ENABLED:
        yield
        return
    _sync()
    t0 = time.perf_counter()
    peak0 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    try:
        yield
    finally:
        _sync()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rec = dict(section=section, seconds=round(dt, 4),
                   peak_vram_gb=round(max(peak, peak0) / 1e9, 3))
        rec.update(meta)
        _TIMING.append(rec)


def timing_reset():
    _TIMING.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def timing_save(out_dir, extra=None):
    if not _TIMING:
        return
    import csv as _csv
    keys = sorted({k for r in _TIMING for k in r} | set((extra or {}).keys()))
    path = os.path.join(out_dir, "timing.csv")
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in _TIMING:
            row = dict(r)
            if extra:
                row.update(extra)
            w.writerow({k: row.get(k, "") for k in keys})
    secs = {}
    for r in _TIMING:
        secs.setdefault(r["section"], []).append(r["seconds"])
    total = sum(sum(v) for v in secs.values())
    print(f"  [timing] total {total:.1f}s  ->  " +
          "  ".join(f"{k} {sum(v):.1f}s ({100*sum(v)/max(total,1e-9):.0f}%)"
                    for k, v in sorted(secs.items(), key=lambda kv: -sum(kv[1]))))
    print(f"  [timing] written {path}")


def timing_report(roots=None):
    import csv as _csv, glob as _glob
    roots = roots or sorted(_glob.glob("*/*/timing.csv"))
    if not roots:
        print("[timing] no timing.csv found -- run with TIMING_ENABLED=True first.")
        return {}
    rows = []
    for p in roots:
        run = p.split(os.sep)[0]
        with open(p) as f:
            for r in _csv.DictReader(f):
                r["run"] = run
                rows.append(r)
    if not rows:
        return {}

    agg = {}
    for r in rows:
        key = (r["run"], r.get("method", "?"), r["section"])
        agg.setdefault(key, []).append(float(r["seconds"]))

    print("=" * 92)
    print("TIMING  --  seconds per step, by stage")
    print("=" * 92)
    print(f"{'run':<30}{'method':<11}{'section':<11}{'n':>4}{'mean s':>9}"
          f"{'total s':>10}{'% of run':>10}")
    run_tot = {}
    for (run, meth, sec), v in agg.items():
        run_tot[run] = run_tot.get(run, 0.0) + sum(v)
    for (run, meth, sec), v in sorted(agg.items()):
        print(f"{run:<30}{meth:<11}{sec:<11}{len(v):>4}{np.mean(v):>9.2f}"
              f"{sum(v):>10.1f}{100*sum(v)/max(run_tot[run],1e-9):>9.0f}%")

    print("\n--- per-run edit cost (the comparable number) ---")
    print(f"{'run':<30}{'method':<11}{'edit steps':>11}{'s / edit':>10}{'run total s':>13}")
    for run in sorted(run_tot):
        ed = [s for (r, m, sec), v in agg.items() if r == run and sec == "edit" for s in v]
        meth = next((m for (r, m, _), _ in agg.items() if r == run), "?")
        if ed:
            print(f"{run:<30}{meth:<11}{len(ed):>11}{np.mean(ed):>10.2f}"
                  f"{run_tot[run]:>13.1f}")
    print("\n[note] latent masking adds one VAE encode per edit plus a blend on every")
    print("       denoise step. Expect a modest overhead, not a doubling -- but")
    print("       measure it rather than asserting it.")
    return agg

#ROI-zoom knobs
EDIT_LONG    = 1280
MAX_UP       = 1.6
ROI_ZOOM     = 1.7       # ROI = object bbox * this
MIN_ROI_FRAC = 0.34      # ROI at least this fraction of each dim
SEED_ROI     = {}
TONE_MATCH_MODE = "boundary"
TONE_BAND_PX    = 24

#detection knobs
ALIGN_GEOMETRY = True
ECC_MAX_ITER   = 50
ECC_EPS        = 1e-4
DIFF_FLOOR     = 0.040
DIFF_K         = 2.5
MIN_AREA_FRAC  = 0.0008
MAX_CHANGE_FRAC= 0.60
MORPH_FRAC     = 0.006
MASK_PAD_FRAC  = 0.02
MASK_FEATHER   = 21
BBOX_PAD_FRAC  = 0.06

# display knobs 
BG_DIM        = 0.18
BG_DESAT      = 0.45
BG_BLUR       = 7
BG_FEATHER_PX = 60
HALO   = (0, 0, 0, 230)
ACCENT = (255, 210, 0, 255)
HALO_LIGHT  = (255, 255, 255, 235)
ACCENT_COOL = (0, 60, 200, 255)
CUE_FIX_HUE = True


def _lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luma(rgb):
    return 0.2126*_lin(rgb[0]) + 0.7152*_lin(rgb[1]) + 0.0722*_lin(rgb[2])


def _contrast(fg, bg):
    l1, l2 = _luma(fg), _luma(bg); hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def choose_palette(disp_arr, bbox):
    H, W = disp_arr.shape[:2]; x0, y0, x1, y1 = [int(v) for v in bbox]
    pad = int(0.15 * max(x1 - x0, y1 - y0)) + 4
    r = disp_arr[max(0, y0-pad):min(H, y1+pad), max(0, x0-pad):min(W, x1+pad)].reshape(-1, 3)
    bg = r.mean(0) if r.size else np.array([128, 128, 128], float)
    o = disp_arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].reshape(-1, 3)
    objc = o.mean(0) if o.size else bg
    if globals().get("CUE_FIX_HUE", True):
        return (ACCENT, HALO if _luma(bg) < 0.5 else HALO_LIGHT)
    warm = min(_contrast(ACCENT[:3], bg), _contrast(ACCENT[:3], objc))
    cool = min(_contrast(ACCENT_COOL[:3], bg), _contrast(ACCENT_COOL[:3], objc))
    return (ACCENT, HALO) if warm >= cool else (ACCENT_COOL, HALO_LIGHT)


def _np(img):   return np.asarray(img.convert("RGB"), dtype=np.float32)
def _np01(img): return _np(img) / 255.0

#  CHANGE DETECTION  (tone/geometry-invariant)
def geometry_align(edit_arr, frame_arr):
    if not ALIGN_GEOMETRY:
        return edit_arr
    try:
        g_ref = cv2.cvtColor(frame_arr.astype(np.float32), cv2.COLOR_RGB2GRAY)
        g_in  = cv2.cvtColor(edit_arr.astype(np.float32),  cv2.COLOR_RGB2GRAY)
        g_ref = (g_ref - g_ref.mean()) / (g_ref.std() + 1e-6)
        g_in  = (g_in  - g_in.mean())  / (g_in.std()  + 1e-6)
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ECC_MAX_ITER, ECC_EPS)
        cv2.findTransformECC(g_ref, g_in, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        H, W = frame_arr.shape[:2]
        return cv2.warpAffine(edit_arr, warp, (W, H),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        return edit_arr


def tone_match(src, ref, exclude=None, fit_mask=None):
    out = src.copy()
    if fit_mask is not None:
        m = fit_mask.astype(bool)
    elif exclude is None:
        m = np.ones(src.shape[:2], bool)
    else:
        m = (exclude == 0)
    if m.sum() < 0.02 * m.size:          # too few samples -> fall back to global
        m = np.ones(src.shape[:2], bool)
    for c in range(3):
        sv, rv = src[..., c][m], ref[..., c][m]
        ss, rs = sv.std() + 1e-6, rv.std() + 1e-6
        out[..., c] = (src[..., c] - sv.mean()) * (rs / ss) + rv.mean()
    return np.clip(out, 0, 255)


def _boundary_ring(change, band=None):
    band = band or TONE_BAND_PX
    k = np.ones((int(band) | 1,) * 2, np.uint8)
    grown = cv2.dilate(change.astype(np.uint8), k)
    return (grown > 0) & (change == 0)


def detect_change(frame_img, edit_img):
    frame = _np(frame_img); edit = _np(edit_img)
    if edit.shape != frame.shape:
        edit_img = edit_img.resize((frame.shape[1], frame.shape[0]), Image.BICUBIC)
        edit = _np(edit_img)

    edit = geometry_align(edit, frame)
    aligned = tone_match(edit, frame, exclude=None)
    rough = np.abs(aligned - frame).mean(2) / 255.0
    rough_change = (rough > max(DIFF_FLOOR, rough.mean() + DIFF_K * rough.std())).astype(np.uint8)
    if TONE_MATCH_MODE == "off":
        aligned = edit.copy()
    elif TONE_MATCH_MODE == "boundary":
        aligned = tone_match(edit, frame, fit_mask=_boundary_ring(rough_change))
    else:
        aligned = tone_match(edit, frame, exclude=rough_change)

    diff = np.abs(aligned - frame).mean(2) / 255.0
    thr = max(DIFF_FLOOR, diff.mean() + DIFF_K * diff.std())
    raw = (diff > thr).astype(np.uint8)

    H, W = raw.shape
    k = max(3, int(MORPH_FRAC * min(H, W)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ker)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, ker)
    if raw.sum() == 0:
        return None, None, aligned.astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    min_area = MIN_AREA_FRAC * H * W
    keep = np.zeros_like(raw)
    for ci in range(1, n):
        if stats[ci, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == ci] = 1
    if keep.sum() == 0:
        ci = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); keep[lab == ci] = 1
    if keep.sum() > MAX_CHANGE_FRAC * H * W:
        ci = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (lab == ci).astype(np.uint8)
        if keep.sum() > MAX_CHANGE_FRAC * H * W:
            return None, None, aligned.astype(np.uint8)

    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
    px, py = int(BBOX_PAD_FRAC * W), int(BBOX_PAD_FRAC * H)
    bbox = (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))
    return keep, bbox, aligned.astype(np.uint8)


def hard_composite(frame_img, edit_toned_arr, keep):
    a = _np(frame_img); b = edit_toned_arr.astype(np.float32)
    m = keep.astype(np.float32)
    pad = int(MASK_PAD_FRAC * max(m.shape)) | 1
    if pad >= 3:
        m = cv2.dilate(m, np.ones((pad, pad), np.uint8))
    if MASK_FEATHER > 0:
        m = cv2.GaussianBlur(m, (MASK_FEATHER | 1, MASK_FEATHER | 1), 0)
    m = m[..., None]
    out = b * m + a * (1 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

#  ROI ZOOM  (scale + object continuity)
def fit_res(size, long_target=EDIT_LONG, mult=16, max_up=MAX_UP):
    w, h = size
    up = min(long_target / float(max(w, h)), max_up)
    nw = max(mult, int(round(w * up / mult)) * mult)
    nh = max(mult, int(round(h * up / mult)) * mult)
    return (nw, nh)


def roi_from_bbox(bbox, W, H, zoom=ROI_ZOOM, minf=MIN_ROI_FRAC):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max((x1 - x0) * zoom, minf * W)
    h = max((y1 - y0) * zoom, minf * H)
    nx0, ny0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    nx1, ny1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    if nx1 - nx0 < 16 or ny1 - ny0 < 16:
        return (0, 0, W, H)
    return (nx0, ny0, nx1, ny1)


def edit_roi(pipe, frame, clean, prompt, roi):
    fcrop = frame.crop(roi); acrop = clean.crop(roi)
    up = fit_res(fcrop.size)
    fup = fcrop.resize(up, Image.LANCZOS); aup = acrop.resize(up, Image.LANCZOS)
    edited_up = edit_once(pipe, fup, aup, prompt)
    edited = edited_up.resize(fcrop.size, Image.LANCZOS)
    return fcrop, edited


#  CUE ROUTER + DISPLAY
VERB_INTENT = {
    "add":    ["place", "insert", "attach", "plug", "fit", "connect", "add",
               "apply", "mount", "hang", "wrap", "tie", "fasten", "press",
               "join", "secure", "stick", "put"],
    "remove": ["remove", "separate", "unplug", "detach", "disassemble",
               "take", "lift", "open", "unscrew", "pull"],
    "region": ["sort", "paint", "clean", "fold", "restock", "sweep", "wipe",
               "sand", "fill", "pour", "water", "spread", "cover", "close"],
    "mark":   ["mark", "inspect", "identify", "measure", "check", "count",
               "grade", "label", "find", "note"],
    "order":  ["sequence", "order", "next", "then", "step", "arrange"],
}
INTENT_PROTO = {
    "add":    "attach, insert, place or add an object into the scene",
    "remove": "take away, detach, unplug or remove an object",
    "region": "work across a whole area or surface, such as clean, paint, fold, pour or close",
    "mark":   "inspect, measure, check, count or label something",
    "order":  "arrange or put items into a numbered sequence",
}
SIM_THRESH = 0.30
_NLP = None; _EMB = None; _PROTO = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download; download("en_core_web_sm")
                _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = False
    return _NLP


def _get_emb():
    global _EMB, _PROTO
    if _EMB is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB = SentenceTransformer("all-MiniLM-L6-v2")
            keys = list(INTENT_PROTO.keys())
            V = _EMB.encode([INTENT_PROTO[k] for k in keys], normalize_embeddings=True)
            _PROTO = (keys, np.asarray(V, np.float32))
        except Exception:
            _EMB = False
    return _EMB


def _verb_candidates(instr):
    nlp = _get_nlp()
    if not nlp:
        return re.findall(r"[a-z]+", instr)
    doc = nlp(instr)
    root  = [t.lemma_ for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
    return root + verbs + re.findall(r"[a-z]+", instr)


def classify_intent(instr):
    instr = (instr or "").lower().strip()
    if not instr:
        return None
    w2i = {v: k for k, vs in VERB_INTENT.items() for v in vs}
    for cand in _verb_candidates(instr):
        if cand in w2i:
            return w2i[cand]
    emb = _get_emb()
    if emb:
        v = emb.encode([instr], normalize_embeddings=True)[0]
        keys, P = _PROTO
        sims = P @ np.asarray(v, np.float32)
        j = int(np.argmax(sims))
        if float(sims[j]) >= SIM_THRESH:
            return keys[j]
    return None


def _area_frac(box, img):
    w = max(0, box[2]-box[0]); h = max(0, box[3]-box[1])
    return (w*h)/float(img.width*img.height)


def pick_cue(intent, box, img):
    small = _area_frac(box, img) < 0.08
    return {"add": "arrow", "remove": "ring", "region": "box",
            "mark": "glow", "order": "badge"}.get(intent, "arrow" if small else "box")


def draw_ring(img, box, accent=ACCENT, halo=HALO, **kw):
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse(box, outline=halo, width=14); d.ellipse(box, outline=accent, width=7); return img


def draw_box(img, box, accent=ACCENT, halo=HALO, **kw):
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle(box, radius=18, outline=halo, width=14)
    d.rounded_rectangle(box, radius=18, outline=accent, width=7); return img


def draw_arrow(img, box, gap=26, accent=ACCENT, halo=HALO, **kw):
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
    sx = float(max(0, box[0]-int(img.width*0.18))); sy = float(max(0, box[1]-int(img.height*0.18)))
    pb = (box[0]-gap, box[1]-gap, box[2]+gap, box[3]+gap); tipx, tipy = cx, cy
    for k in range(241):
        t = k/240; px = sx+t*(cx-sx); py = sy+t*(cy-sy)
        if pb[0] <= px <= pb[2] and pb[1] <= py <= pb[3]:
            tipx, tipy = px, py; break
    ang = math.atan2(tipy-sy, tipx-sx)
    for col, w in [(halo, 18), (accent, 9)]:
        d.line([(sx, sy), (tipx, tipy)], fill=col, width=w)
    for col, w in [(halo, 18), (accent, 9)]:
        for da in (math.radians(150), math.radians(-150)):
            d.line([(tipx, tipy), (tipx+42*math.cos(ang+da), tipy+42*math.sin(ang+da))], fill=col, width=w)
    return img


def draw_glow(img, box, accent=ACCENT, halo=HALO, **kw):
    cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
    rad = int(0.6*max(box[2]-box[0], box[3]-box[1]))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    for k in range(10, 0, -1):
        rr = int(rad*k/10); a = int(120*(1-k/11))
        gd.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], fill=(accent[0], accent[1], accent[2], a))
    out = Image.alpha_composite(img.convert("RGBA"), glow)
    ImageDraw.Draw(out, "RGBA").ellipse(box, outline=accent, width=5); return out.convert("RGB")


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


def draw_badge(img, box, step=None, accent=ACCENT, halo=HALO, **kw):
    img = draw_ring(img.copy(), box, accent=accent, halo=halo); d = ImageDraw.Draw(img, "RGBA")
    R = max(22, int(0.045*min(img.width, img.height))); bx, by = box[0], box[1]
    d.ellipse([bx-R, by-R, bx+R, by+R], fill=halo)
    d.ellipse([bx-R+4, by-R+4, bx+R-4, by+R-4], fill=accent)
    txt = str(step) if step is not None else "!"
    f = _font(int(R*1.1)); tb = d.textbbox((0, 0), txt, font=f)
    d.text((bx-(tb[2]-tb[0])/2, by-(tb[3]-tb[1])/2-tb[1]), txt, fill=(0, 0, 0, 255), font=f); return img


DRAW = {"ring": draw_ring, "box": draw_box, "arrow": draw_arrow,
        "glow": draw_glow, "badge": draw_badge}


def reduce_background(img, box):
    rgb = img.convert("RGB"); arr = np.asarray(rgb, np.float32); calm = arr.copy()
    if BG_BLUR > 0:
        calm = np.asarray(rgb.filter(ImageFilter.GaussianBlur(BG_BLUR)), np.float32)
    gray = calm.mean(axis=2, keepdims=True)
    calm = calm*(1-BG_DESAT) + gray*BG_DESAT; calm = calm*(1-BG_DIM)
    H, W = arr.shape[:2]; mask = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = box; mask[y0:y1, x0:x1] = 1.0
    if BG_FEATHER_PX > 0:
        mask = cv2.GaussianBlur(mask, (BG_FEATHER_PX | 1, BG_FEATHER_PX | 1), 0)
    mask = mask[..., None]; out = arr*mask + calm*(1-mask)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

CUE_FORCE_BOX = True
CUE_ALWAYS = True
def _loose_change_bbox(before, after, offset=(0, 0), thresh=8, min_frac=0.0004):
    import cv2
    a = np.asarray(before.convert("L"), np.int16)
    b = np.asarray(after.convert("L"), np.int16)
    d = (np.abs(a - b) > thresh).astype(np.uint8)
    if d.mean() < min_frac:
        return None
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.where(d > 0)
    if xs.size == 0:
        return None
    ox, oy = offset
    return (int(xs.min() + ox), int(ys.min() + oy),
            int(xs.max() + ox), int(ys.max() + oy))


def make_display(clean_frame, bbox, step, instructions=None):
    if bbox is None:
        return clean_frame.copy()
    instrs = instructions if instructions is not None else INSTRUCTIONS
    intent = classify_intent(instrs[step-1] if step-1 < len(instrs) else "")
    cue = pick_cue(intent, bbox, clean_frame)
    calmed = reduce_background(clean_frame, bbox)
    accent, halo = choose_palette(np.asarray(calmed, np.float32), bbox)
    out = DRAW[cue](calmed.copy(), bbox, step=step, accent=accent, halo=halo)
    if globals().get("CUE_FORCE_BOX", False) and cue != "box":
        out = draw_box(out, bbox, accent=accent, halo=halo)   # enclose the changed region
    return out

import builtins, gc

_PIPE_KEY   = "_QWEN2511_PIPE"
_ESRGAN_KEY = "_QWEN2511_ESRGAN"


def _vram(tag=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[vram]{(' ' + tag) if tag else ''} allocated={a:.2f} GB reserved={r:.2f} GB")


def get_pipe(force_reload=False):
    if getattr(builtins, "_QWENIMAGE_T2I_PIPE", None) is not None:
        raise RuntimeError(
            "[pipe] The Cell 0 text-to-image model is STILL LOADED. Call "
            "free_gen_pipe() first, then get_pipe(). (Loading both = CUDA OOM.)")
    cached = getattr(builtins, _PIPE_KEY, None)
    if cached is not None and not force_reload:
        print("[pipe] reusing already-loaded pipeline (no reload, no extra VRAM)")
        return cached
    if cached is not None and force_reload:
        free_pipe()
    print(f"[pipe] loading {MODEL_ID} ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    setattr(builtins, _PIPE_KEY, pipe)
    _vram("after load")
    return pipe


def free_pipe(verbose=True):
    global _ESRGAN
    freed = False
    for key in (_PIPE_KEY, _ESRGAN_KEY):
        obj = getattr(builtins, key, None)
        if obj is not None:
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
            try:
                delattr(builtins, key)
            except Exception:
                pass
            del obj
            freed = True
    _ESRGAN = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if verbose:
        print("[pipe] unloaded." if freed else "[pipe] nothing loaded.")
        _vram("after free")


def load_pipe():
    return get_pipe()


#  NULL-PASS MANIFOLD PROJECTION
def project_to_manifold(pipe, real):
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    prompt = ("Using the first image as the scene and the second as the reference, "
              "keep the room, the walls, the floor, the lighting, all furniture and "
              "equipment, any people already present, and the background exactly as "
              "they appear: " + NULL_INSTRUCTION)
    with torch.inference_mode():
        out = pipe(image=[real, real], prompt=prompt, negative_prompt=NEG,
                   true_cfg_scale=NULL_CFG, num_inference_steps=NULL_STEPS,
                   guidance_scale=1.0, num_images_per_prompt=1, generator=g).images[0]
    if out.size != real.size:
        out = out.resize(real.size, Image.LANCZOS)
    return out

#  DISPLAY QUALITY UPGRADE
ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
              "v0.1.0/RealESRGAN_x4plus.pth")


def _ensure_weights(path=None, url=ESRGAN_URL):
    path = path or ESRGAN_WEIGHTS
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    print(f"[enhance] weights missing -> downloading Real-ESRGAN ({url})")
    import urllib.request
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"[enhance] weights ready: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


_ESRGAN = None


def _get_esrgan():
    global _ESRGAN
    if _ESRGAN is None:
        _ESRGAN = getattr(builtins, _ESRGAN_KEY, None)
    if _ESRGAN is not None:
        return _ESRGAN
    try:
        import sys, types
        import torchvision.transforms.functional as _tvF
        if "torchvision.transforms.functional_tensor" not in sys.modules:
            _shim = types.ModuleType("torchvision.transforms.functional_tensor")
            _shim.rgb_to_grayscale = _tvF.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = _shim
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        wpath = _ensure_weights()
        _ESRGAN = RealESRGANer(scale=4, model_path=wpath, model=model,
                               tile=512, tile_pad=10, half=True, device=DEVICE)
        setattr(builtins, _ESRGAN_KEY, _ESRGAN)
        print(f"[enhance] Real-ESRGAN LOADED (x4 model, outscale={ENHANCE_SCALE}).")
    except Exception as e:
        print(f"[enhance] Real-ESRGAN unavailable ({type(e).__name__}: {e})")
        print("[enhance] -> using deterministic LANCZOS+unsharp fallback "
              "(cosmetic only; pipeline results are unaffected).")
        _ESRGAN = False
    return _ESRGAN


def enhance_display(img, scale=ENHANCE_SCALE):
    if not ENHANCE_DISPLAY or scale <= 1:
        return img
    er = _get_esrgan()
    if er:
        arr = np.asarray(img.convert("RGB"))
        out, _ = er.enhance(arr, outscale=scale)
        return Image.fromarray(out)
    w, h = img.size
    up = img.convert("RGB").resize((w * scale, h * scale), Image.LANCZOS)
    return up.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

ANCHOR = ("Using the first image as the scene to edit and the second image as the "
          "reference: keep the room, the walls, the floor, the lighting, all "
          "furniture and equipment, any people already present and their pose and "
          "clothing, and the entire background exactly as they appear in the "
          "reference. Do not add, remove or move anything that the instruction "
          "does not ask for. The instruction is: ")


def edit_once(pipe, frame, clean, prompt):
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    prompt = ANCHOR + prompt
    with torch.inference_mode():
        out = pipe(image=[frame, clean], prompt=prompt, negative_prompt=NEG,
                   true_cfg_scale=CFG, num_inference_steps=STEPS,
                   guidance_scale=1.0, num_images_per_prompt=1,
                   generator=g).images[0]
    if out.size != frame.size:
        out = out.resize(frame.size, Image.LANCZOS)
    return out


def _union(a, b):
    if a is None: return b
    if b is None: return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def contact_sheet(frames, labels, cell=460):
    n = len(frames); sheet = Image.new("RGB", (cell*n, cell+24), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        t = f.convert("RGB").copy(); t.thumbnail((cell, cell))
        sheet.paste(t, (i*cell, 24)); d.text((i*cell+4, 4), labels[i], fill=(255, 255, 255))
    return sheet

CUE_PAD_FRAC    = 0.035   # breathing room on every side, fraction of min(H,W)
CUE_MIN_FRAC    = 0.09    # minimum cue-box side, fraction of min(H,W)
SOFT_COMPOSITE  = False   # True -> feathered paste + contact shadow
FEATHER_PX      = 6
SHADOW_STRENGTH = 0.28


def _pad_cue_bbox(bbox, W, H):
    """Grow a cue bbox outward from the SAME centre, with a minimum visible size
    so a tiny change still gets a box a person can see."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = float(min(W, H))
    bw = max((x1 - x0) + 2.0 * CUE_PAD_FRAC * s, CUE_MIN_FRAC * s)
    bh = max((y1 - y0) + 2.0 * CUE_PAD_FRAC * s, CUE_MIN_FRAC * s)
    return (int(max(0, cx - bw / 2.0)), int(max(0, cy - bh / 2.0)),
            int(min(W, cx + bw / 2.0)), int(min(H, cy + bh / 2.0)))



print("=" * 74)
print("[cell01 v15.0] ENGINE ready (config, Qwen pipe, detect/composite, cue, timing)")
print(f"  TASK_NAME={TASK_NAME!r}  {len(TASK_STEPS)} steps  CFG={CFG}  STEPS={STEPS}")
print(f"  CUE_FIX_HUE={CUE_FIX_HUE}  SOFT_COMPOSITE={SOFT_COMPOSITE}")
print("  THE RUNNERS LIVE IN cell06. This file no longer defines run()/run_one().")
print("  Instructions live HERE and ONLY here: edit TASK_NAME + TASK_STEPS above.")
print("=" * 74)