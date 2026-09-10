#SCS score
import os, csv, glob
import numpy as np
import cv2
from PIL import Image

METRICS_ROOT = "full_2511"
SCORE_ONLY = []

W = {
    "localization": 0.35,   # exactly one compact change  (DISCRIMINATIVE)
    "readability":  0.30,   # cue contrast vs local bg     (DISCRIMINATIVE)
    "stability":    0.20,   # background frozen            (verification)
    "alignment":    0.15,   # cue sits on the change       (verification)
}
INCLUDE_VLM_IN_SCS = False

GATE_NOOP_STEPS = True

METRIC_MIN_AREA_FRAC = 0.0008
ACCENT_RGB    = (255, 210, 0)    
ACCENT_COOL   = (0, 60, 200)     
CUE_TOL       = 60                
CUE_MIN_FRAC  = 0.003             

def _lum(rgb):
    """WCAG relative luminance from sRGB 0..255."""
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def wcag_contrast(fg_rgb, bg_rgb):
    l1, l2 = float(_lum(np.array(fg_rgb))), float(_lum(np.array(bg_rgb)))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32)

def m_localization(mask):
    m = (mask > 127).astype(np.uint8)
    H, Wd = m.shape
    if m.sum() == 0:
        return 0.0, 0, 0.0, "no_change"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    all_areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    raw_max = float(all_areas.max()) / (H * Wd) if all_areas.size else 0.0
    min_area = METRIC_MIN_AREA_FRAC * H * Wd
    comps = [ci for ci in range(1, n) if stats[ci, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return 0.0, 0, raw_max, "below_floor"
    areas = np.array([stats[ci, cv2.CC_STAT_AREA] for ci in comps], float)
    big = comps[int(np.argmax(areas))]
    concentration = areas.max() / areas.sum()
    bw, bh = stats[big, cv2.CC_STAT_WIDTH], stats[big, cv2.CC_STAT_HEIGHT]
    compact = stats[big, cv2.CC_STAT_AREA] / float(max(1, bw * bh))
    score = concentration * (0.5 + 0.5 * compact)
    return float(np.clip(score, 0, 1)), len(comps), float(areas.max() / (H * Wd)), "ok"


def m_stability(prev_clean, clean, mask):
    m = (mask > 127)
    outside = ~m
    if outside.sum() == 0:
        return 1.0
    resid = np.abs(prev_clean[outside] - clean[outside]).mean() / 255.0
    return float(np.clip(1.0 - resid * 4.0, 0, 1))


def _cue_pixels(display, bbox, mask_shape, search_bbox=None):
    H, Wd = display.shape[:2]
    sx = Wd / float(mask_shape[1]); sy = H / float(mask_shape[0])
    x0, y0, x1, y1 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
    if search_bbox is not None:
        rx0 = max(0, int(search_bbox[0]*sx) - 12); ry0 = max(0, int(search_bbox[1]*sy) - 12)
        rx1 = min(Wd, int(search_bbox[2]*sx) + 12); ry1 = min(H, int(search_bbox[3]*sy) + 12)
        reg = display[ry0:ry1, rx0:rx1].astype(np.float32)
        if reg.size == 0:
            return np.array([]), np.array([]), (x0, y0, x1, y1)
        dW = np.linalg.norm(reg - np.array(ACCENT_RGB, np.float32), axis=2)
        dC = np.linalg.norm(reg - np.array(ACCENT_COOL, np.float32), axis=2)
        ys, xs = np.where((dW < CUE_TOL) | (dC < CUE_TOL))
        return ys + ry0, xs + rx0, (x0, y0, x1, y1)
    ow, oh = max(1, x1 - x0), max(1, y1 - y0); pad = int(0.35 * max(ow, oh)) + 12
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(Wd, x1 + pad), min(H, y1 + pad)
    reg = display[ry0:ry1, rx0:rx1].astype(np.float32)
    if reg.size == 0:
        return np.array([]), np.array([]), (x0, y0, x1, y1)
    dW = np.linalg.norm(reg - np.array(ACCENT_RGB, np.float32), axis=2)
    dC = np.linalg.norm(reg - np.array(ACCENT_COOL, np.float32), axis=2)
    ys, xs = np.where((dW < CUE_TOL) | (dC < CUE_TOL))
    return ys + ry0, xs + rx0, (x0, y0, x1, y1)


def m_alignment(display, mask, bbox, mask_shape, cue_bbox=None):
    if bbox is None:
        return 0.0
    ys, xs, sbb = _cue_pixels(display, bbox, mask_shape, search_bbox=cue_bbox)
    if xs.size < 20:
        return 0.0                                  
    ccx, ccy = xs.mean(), ys.mean()
    chx, chy = (sbb[0] + sbb[2]) / 2.0, (sbb[1] + sbb[3]) / 2.0
    diag = np.hypot(display.shape[0], display.shape[1])
    dist = np.hypot(ccx - chx, ccy - chy) / diag
    return float(np.clip(1.0 - dist * 6.0, 0, 1))


def m_readability(display, bbox, mask_shape, cue_bbox=None):
    if cue_bbox is not None and bbox is not None:
        H, Wd = display.shape[:2]
        sx = Wd / float(mask_shape[1]); sy = H / float(mask_shape[0])
        cx0, cy0, cx1, cy1 = (int(cue_bbox[0]*sx), int(cue_bbox[1]*sy),
                              int(cue_bbox[2]*sx), int(cue_bbox[3]*sy))
        ox0, oy0, ox1, oy1 = (int(bbox[0]*sx), int(bbox[1]*sy),
                              int(bbox[2]*sx), int(bbox[3]*sy))
        mrg = int(0.10 * max(cx1 - cx0, cy1 - cy0)) + 6
        rx0, ry0 = max(0, cx0 - mrg), max(0, cy0 - mrg)
        rx1, ry1 = min(Wd, cx1 + mrg), min(H, cy1 + mrg)
        if rx1 <= rx0 or ry1 <= ry0:
            return 0.0
        yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
        ann = ~((xx >= ox0) & (xx < ox1) & (yy >= oy0) & (yy < oy1))
        ap = display[ry0:ry1, rx0:rx1].reshape(-1, 3).astype(np.float32)[ann.reshape(-1)]
        if ap.shape[0] < 20:
            return 0.0
        d_warm = np.linalg.norm(ap - np.array(ACCENT_RGB, np.float32), axis=1)
        d_cool = np.linalg.norm(ap - np.array(ACCENT_COOL, np.float32), axis=1)
        cue = (d_warm < CUE_TOL) | (d_cool < CUE_TOL)
        if float(cue.mean()) < CUE_MIN_FRAC:
            return 0.0
        cue_color = ap[cue].mean(0)
        bg = ap[~cue].mean(0) if (~cue).sum() > 10 else np.array([128, 128, 128], np.float32)
        ratio = wcag_contrast(cue_color, bg)
        return float(np.clip((ratio - 1.0) / (4.5 - 1.0), 0, 1))
    if bbox is None:
        return 0.0
    H, Wd = display.shape[:2]
    sx = Wd / float(mask_shape[1]); sy = H / float(mask_shape[0])
    x0, y0, x1, y1 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
    ow, oh = max(1, x1 - x0), max(1, y1 - y0)
    pad = int(0.30 * max(ow, oh)) + 10
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(Wd, x1 + pad), min(H, y1 + pad)
    if rx1 <= rx0 or ry1 <= ry0:
        return 0.0
    yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
    ann = ~((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))
    ap = display[ry0:ry1, rx0:rx1].reshape(-1, 3).astype(np.float32)[ann.reshape(-1)]
    if ap.shape[0] < 20:
        return 0.0
    d_warm = np.linalg.norm(ap - np.array(ACCENT_RGB, np.float32), axis=1)
    d_cool = np.linalg.norm(ap - np.array(ACCENT_COOL, np.float32), axis=1)
    cue = (d_warm < CUE_TOL) | (d_cool < CUE_TOL)
    if float(cue.mean()) < CUE_MIN_FRAC:
        return 0.0                                              
    cue_color = ap[cue].mean(0)
    bg = ap[~cue].mean(0) if (~cue).sum() > 10 else np.array([128, 128, 128], np.float32)
    ratio = wcag_contrast(cue_color, bg)
    return float(np.clip((ratio - 1.0) / (4.5 - 1.0), 0, 1))

def bbox_from_mask(mask):
    m = (mask > 127).astype(np.uint8)
    if m.sum() == 0:
        return None
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def score_stem(stem, instructions, root=METRICS_ROOT):
    d = os.path.join(root, stem)
    rows, scores = [], []
    inp = os.path.join(d, "step_00_input.png")
    if not os.path.exists(inp):
        return rows, 0.0
    steps = sorted(glob.glob(os.path.join(d, "step_*_clean.png")))
    prev_clean = load(inp)
    for p in steps:
        i = int(os.path.basename(p).split("_")[1])
        clean = load(p)
        mpath = os.path.join(d, f"step_{i:02d}_mask.png")
        dpath = os.path.join(d, f"step_{i:02d}_display.png")
        if not os.path.exists(mpath):
            prev_clean = clean; continue
        mask = np.asarray(Image.open(mpath).convert("L"))
        cpath = os.path.join(d, f"step_{i:02d}_cue.png")
        if os.path.exists(cpath):
            display = load(cpath)
        elif os.path.exists(dpath):
            display = load(dpath)
        else:
            display = clean
        bbox = bbox_from_mask(mask)

        cue_bbox = None
        cbpath = os.path.join(d, f"step_{i:02d}_cuebbox.txt")
        if os.path.exists(cbpath):
            try:
                cue_bbox = tuple(int(float(v)) for v in open(cbpath).read().split(","))
            except Exception:
                cue_bbox = None

        loc, ncomp, area, loc_kind = m_localization(mask)
        stab = m_stability(prev_clean, clean, mask)
        alg  = m_alignment(display, mask, bbox, mask.shape, cue_bbox=cue_bbox)
        read = m_readability(display, bbox, mask.shape, cue_bbox=cue_bbox)
        instr = instructions[i - 1] if i - 1 < len(instructions) else ""

        vlm = vlm_recoverability(os.path.join(d, "step_%02d_clean.png" % (i - 1)) if i > 1
                                 else os.path.join(d, "step_00_input.png"), p, instr)

        scs_geom = (W["localization"] * loc + W["readability"] * read +
                    W["stability"] * stab + W["alignment"] * alg)

        noop = (loc_kind == "no_change")
        scs_gated = 0.0 if noop else scs
        scs_geom_gated = 0.0 if noop else scs_geom

        rows.append(dict(step=i, n_components=ncomp, change_area=round(area, 4),
                         loc_kind=loc_kind,
                         localization=round(loc, 3), readability=round(read, 3),
                         stability=round(stab, 3), alignment=round(alg, 3),
                         SCS=round(scs, 3), SCS_geom=round(scs_geom, 3),
                         noop=int(noop),
                         SCS_gated=round(scs_gated, 3),
                         SCS_geom_gated=round(scs_geom_gated, 3),
                         vlm_recoverability=("" if vlm is None else round(vlm, 3)),
                         instruction=instr))
        scores.append(scs)
        prev_clean = clean

    if rows:
        with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
    return rows, (float(np.mean(scores)) if scores else 0.0)


def _instructions_for(root):
    variants = globals().get("VARIANTS", None)
    if not variants:
        return []
    low = root.lower()
    for key in sorted(variants.keys(), key=len, reverse=True):
        if key in low:
            return variants[key]
    return variants.get("mat", next(iter(variants.values())))


def score_all():
    if SCORE_ONLY:
        roots = [r for r in SCORE_ONLY if os.path.isdir(r)]
        missing = [r for r in SCORE_ONLY if not os.path.isdir(r)]
        if missing:
            print(f"[warn] SCORE_ONLY folders not found (skipped): {missing}")
    else:
        roots = sorted([r for r in glob.glob("full_2511*") if os.path.isdir(r)])
    if not roots:
        raise SystemExit("[err] no folders to score -- run Cell 1 first, or fix SCORE_ONLY.")
    print(f"[scoring folders] {roots}")
    print(f"{'variant':<26}{'image':<16}{'step':>4}  {'loc':>5}{'read':>6}{'stab':>6}"
          f"{'algn':>6}{'SCS':>6}{'gated':>7}{'vlm':>6}  instruction")
    summary = {}
    vlm_sums = {}
    gated_sums = {}
    compl_sums = {}
    for root in roots:
        instructions = _instructions_for(root)
        stems = sorted([s for s in os.listdir(root)
                        if os.path.isdir(os.path.join(root, s))
                        and not s.startswith(".")
                        and os.path.exists(os.path.join(root, s, "step_00_input.png"))])
        for stem in stems:
            rows, avg = score_stem(stem, instructions, root=root)
            vlm_vals = []
            for r in rows:
                v = r.get("vlm_recoverability", "")
                vlm_str = f"{v:>6}" if v != "" else f"{'--':>6}"
                if v != "":
                    vlm_vals.append(float(v))
                gcell = f"{r.get('SCS_gated', r['SCS']):>7}"
                dead = "  [NO-OP]" if r.get("noop") else ""
                print(f"{root:<26}{stem:<16}{r['step']:>4}  {r['localization']:>5}"
                      f"{r['readability']:>6}{r['stability']:>6}{r['alignment']:>6}"
                      f"{r['SCS']:>6}{gcell}{vlm_str}  {r['instruction'][:34]}{dead}")
            vlm_avg = (sum(vlm_vals) / len(vlm_vals)) if vlm_vals else None
            vlm_cell = f"{vlm_avg:>6.3f}" if vlm_avg is not None else f"{'--':>6}"
            g_vals = [r.get("SCS_gated", r["SCS"]) for r in rows]
            gated_avg = float(np.mean(g_vals)) if g_vals else 0.0
            n_dead = sum(1 for r in rows if r.get("noop"))
            n_small = sum(1 for r in rows if r.get("loc_kind") == "below_floor")
            completion = (1.0 - n_dead / len(rows)) if rows else 0.0
            print(f"{root:<26}{stem:<16}{'AVG':>4}  {'':>5}{'':>6}{'':>6}{'':>6}"
                  f"{avg:>6.3f}{gated_avg:>7.3f}{vlm_cell}  "
                  f"completion {100*completion:.0f}% ({len(rows)-n_dead}/{len(rows)})"
                  f"   below-floor {n_small}")
            summary[(root, stem)] = avg
            vlm_sums[(root, stem)] = vlm_avg
            gated_sums[(root, stem)] = gated_avg
            compl_sums[(root, stem)] = completion
    for (root, stem), avg in summary.items():
        v = vlm_sums.get((root, stem))
        vtxt = f"{v:.3f}" if v is not None else "--"
        g = gated_sums.get((root, stem), avg)
        c = compl_sums.get((root, stem), 1.0)
        print(f"  {root:<26}{stem:<16}SCS={avg:.3f}   gated={g:.3f}   "
              f"completion={100*c:>3.0f}%   vlm={vtxt}")
    return summary