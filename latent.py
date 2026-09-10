#latent
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None

VAE_DOWN     = 8
PATCH        = 2
TOKEN_STRIDE = VAE_DOWN * PATCH

MASK_DILATE_FRAC  = 0.04
MASK_FEATHER_FRAC = 0.02
DEBUG       = True
STRICT_MASK = True        
SHARE_INIT_NOISE = True
_SHARE_OK = None
BLEND_STOP_FRAC = 0.75


def _log(*a):
    if DEBUG:
        print("[latent-mask]", *a)


def _exec_device(pipe):
    for cand in (getattr(pipe, "_execution_device", None), getattr(pipe, "device", None)):
        if cand is None:
            continue
        d = torch.device(cand)
        if d.type not in ("meta",):
            return d
    return torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
def build_token_mask(mask_hw, H, W, dilate=MASK_DILATE_FRAC, feather=MASK_FEATHER_FRAC):
    import cv2
    h2, w2 = H // TOKEN_STRIDE, W // TOKEN_STRIDE
    m = np.asarray(mask_hw).astype(np.float32)
    if m.max() > 1.5:
        m /= 255.0
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    if dilate > 0:
        k = max(3, int(dilate * min(H, W)) | 1)
        m = cv2.dilate(m, np.ones((k, k), np.uint8))

    # downsample to the TOKEN grid (area-average keeps partial coverage as a soft value)
    m = cv2.resize(m, (w2, h2), interpolation=cv2.INTER_AREA)

    if feather > 0:
        k = max(3, int(feather * min(h2, w2)) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)

    m = np.clip(m, 0.0, 1.0).reshape(1, h2 * w2, 1)     # row-major == token order
    _log(f"token mask {h2}x{w2} = {h2*w2} tokens | editable {float(m.mean())*100:.1f}%")
    return m

def _pil_to_tensor(pipe, img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]           # (1,3,H,W)
    t = t.unsqueeze(2)                                          # (1,3,1,H,W)
    return t.to(device=_exec_device(pipe), dtype=pipe.vae.dtype).contiguous()


def encode_reference(pipe, img, generator=None):
    x = _pil_to_tensor(pipe, img)
    H, W = img.size[1], img.size[0]
    lat = None
    if hasattr(pipe, "_encode_vae_image"):
        try:
            lat = pipe._encode_vae_image(image=x, generator=generator)
        except TypeError:
            try:
                lat = pipe._encode_vae_image(x, generator)
            except Exception:
                lat = None
        except Exception:
            lat = None
    if lat is None:
        with torch.no_grad():
            enc = pipe.vae.encode(x)
        lat = enc.latent_dist.sample(generator) if hasattr(enc, "latent_dist") else enc.latents
        sf = getattr(pipe.vae.config, "scaling_factor", None)
        sm = getattr(pipe.vae.config, "shift_factor", None)
        if sm is not None:
            lat = lat - sm
        if sf:
            lat = lat * sf

    # Qwen/Wan VAE may return a temporal axis (B, C, T, h, w) -> drop T
    if lat.ndim == 5:
        lat = lat[:, :, 0]
    B, C, h, w = lat.shape
    _log(f"vae latent {tuple(lat.shape)} (expect h={H//VAE_DOWN}, w={W//VAE_DOWN})")
    if hasattr(pipe, "_pack_latents"):
        try:
            packed = pipe._pack_latents(lat, B, C, h, w)
            _log(f"packed via pipeline helper -> {tuple(packed.shape)}")
            return packed
        except Exception as e:
            _log(f"pipeline _pack_latents failed ({e}); using manual pack")
    packed = (lat.view(B, C, h // PATCH, PATCH, w // PATCH, PATCH)
                 .permute(0, 2, 4, 1, 3, 5)
                 .reshape(B, (h // PATCH) * (w // PATCH), C * PATCH * PATCH))
    _log(f"packed manually -> {tuple(packed.shape)}")
    return packed

def make_blend_callback(z_ref, token_mask, eps):
    m = token_mask           
    state = {"applied": 0, "skipped": 0, "no_sigmas": 0, "released": 0,
             "first_sigma": None, "last_sigma": None, "release_at": None,
             "sched_final_sigma": None}

    def cb(pipe, step, timestep, kw):
        z = kw.get("latents", None)
        if z is None:
            return kw
        sigmas = getattr(pipe.scheduler, "sigmas", None)
        if sigmas is None or len(sigmas) == 0:
            state["no_sigmas"] += 1
            return kw
        n_steps = max(1, len(sigmas) - 1)
        state["sched_final_sigma"] = float(sigmas[-1])
        if BLEND_STOP_FRAC < 1.0:
            stop = int(round(BLEND_STOP_FRAC * n_steps))
            state["release_at"] = stop
            if int(step) >= stop:
                state["released"] += 1
                return kw
        j = min(int(step) + 1, len(sigmas) - 1)
        sigma = float(sigmas[j])
        if state["first_sigma"] is None:
            state["first_sigma"] = sigma
        state["last_sigma"] = sigma

        n = m.shape[1]
        if z.shape[1] < n:
            state["skipped"] += 1
            return kw

        ref_t = (1.0 - sigma) * z_ref + sigma * eps
        mm = m.to(device=z.device, dtype=torch.float32)

        blended = z.clone()
        head32 = z[:, :n].to(torch.float32)
        mixed  = mm * head32 + (1.0 - mm) * ref_t[:, :n].to(device=z.device)
        blended[:, :n] = mixed.to(z.dtype)

        state["applied"] += 1
        kw["latents"] = blended
        return kw

    cb.state = state
    return cb
def masked_edit(pipe, frame, clean, prompt, mask=None,
                cfg=4.0, steps=40, seed=1234, neg=" ",
                anchor=True, height=None, width=None, anchor_text=None):
    if torch is None:
        raise RuntimeError("torch unavailable")

    W, H = frame.size
    height = height or H
    width = width or W
    if height % TOKEN_STRIDE or width % TOKEN_STRIDE:
        raise ValueError(f"H/W must be multiples of {TOKEN_STRIDE}; got {height}x{width}")

    if anchor:
        a = anchor_text
        if a is None:
            a = globals().get("ANCHOR", None)
        if a is None:
            a = ("Using the first image as the scene to edit and the second image "
                 "as the reference for keeping the room, walls, workbench, shelves "
                 "and background identical: ")
        prompt = a + prompt

    dev = _exec_device(pipe)
    g = torch.Generator(device=dev).manual_seed(seed)
    call = dict(image=[frame, clean], prompt=prompt, negative_prompt=neg,
                true_cfg_scale=cfg, num_inference_steps=steps,
                guidance_scale=1.0, num_images_per_prompt=1,
                height=height, width=width, generator=g)

    cb = None
    shared = False
    if mask is not None:
        z_ref = encode_reference(pipe, frame, generator=g).to(torch.float32)
        tm = build_token_mask(mask, height, width)
        token_mask = torch.from_numpy(tm).to(device=z_ref.device, dtype=torch.float32)
        eps = torch.randn(z_ref.shape, generator=g,
                          device=z_ref.device, dtype=torch.float32)

        if not hasattr(pipe.scheduler, "sigmas"):
            raise RuntimeError("scheduler has no `sigmas` attribute -- is this a "
                               "flow-matching sampler? Latent masking assumes one.")

        global _SHARE_OK
        if SHARE_INIT_NOISE and _SHARE_OK is not False:
            import inspect
            try:
                params = inspect.signature(pipe.__call__).parameters
            except (TypeError, ValueError):
                params = {}
            if "latents" in params:
                ldtype = getattr(getattr(pipe, "transformer", None), "dtype", None)
                call["latents"] = eps.to(dtype=ldtype or z_ref.dtype)
                shared = True
            elif _SHARE_OK is None:
                _SHARE_OK = False
                print("[latent-mask] pipeline does not accept `latents=` -- falling "
                      "back to independent noise. The feather band will average two "
                      "different trajectories; keep MASK_FEATHER_FRAC small.")

        cb = make_blend_callback(z_ref, token_mask, eps)
        call["callback_on_step_end"] = cb
        call["callback_on_step_end_tensor_inputs"] = ["latents"]

    try:
        with torch.inference_mode():
            out = pipe(**call).images[0]
        if shared:
            _SHARE_OK = True
    except Exception as e:
        if not shared:
            raise
        _SHARE_OK = False
        call.pop("latents", None)
        print(f"[latent-mask] `latents=` rejected ({type(e).__name__}: {e}); "
              f"retrying with the pipeline's own noise (this run only).")
        with torch.inference_mode():
            out = pipe(**call).images[0]

    if cb is not None:
        st = cb.state
        _log(f"blend applied on {st['applied']} steps, released {st['released']} "
             f"(from step {st['release_at']}), skipped {st['skipped']}, "
             f"no-sigma {st['no_sigmas']} | sigma {st['first_sigma']} -> {st['last_sigma']} "
             f"| shared_noise={shared}")
        if st["applied"] == 0:
            msg = ("[latent-mask] callback never fired or the token layout "
                   "mismatched -- this was an UNCONSTRAINED edit. Run diagnose() "
                   "and check the printed token counts.")
            if STRICT_MASK:
                raise RuntimeError(msg + "  (STRICT_MASK=True)")
            print("WARNING: " + msg)
        elif (st["sched_final_sigma"] is not None
              and st["sched_final_sigma"] > 0.05):
            print(f"WARNING: the SCHEDULE ends at sigma = {st['sched_final_sigma']:.4f}, "
                  f"not ~0. The decoded image really will carry noise. Check the "
                  f"scheduler's sigma/shift configuration.")
        elif st["released"] > 0:
            _log(f"clamp released at sigma {st['last_sigma']:.3f} for the final "
                 f"{st['released']} steps; schedule still ended at "
                 f"{st['sched_final_sigma']:.3f}. Output is fully denoised.")

    if out.size != frame.size:
        out = out.resize(frame.size, Image.LANCZOS)
    return out


PROBE_STEPS      = 10       
PROBE_SEEDS      = (1234,)  
PROBE_QUANTILE   = 0.70     
PROBE_SMOOTH_TOK = 3        
PROBE_MIN_FRAC   = 0.03    
PROBE_MAX_FRAC   = 0.75
PROBE_NULL       = "Return the exact same image with no changes at all."


def _probe_trajectory(pipe, frame, clean, full_prompt, eps, steps, cfg, neg,
                      height, width, seed):
    """Run a short denoise and capture the packed latents after every step."""
    caps = []

    def cb(p, step, t, kw):
        z = kw.get("latents")
        if z is not None:
            caps.append(z.detach().to(torch.float32).clone())
        return kw

    dev = _exec_device(pipe)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    call = dict(image=[frame, clean], prompt=full_prompt, negative_prompt=neg,
                true_cfg_scale=cfg, num_inference_steps=int(steps),
                guidance_scale=1.0, num_images_per_prompt=1,
                height=height, width=width, generator=g,
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"])
    if eps is not None:
        ldtype = getattr(getattr(pipe, "transformer", None), "dtype", None)
        call["latents"] = eps.to(dtype=ldtype or torch.float32)
    with torch.inference_mode():
        pipe(**call)
    return caps


def derive_instruction_mask(pipe, frame, clean, instruction, neg=" ", cfg=4.0,
                            height=None, width=None, anchor_text=None,
                            steps=None, seeds=None, quantile=None,
                            save_heatmap=None):
    if torch is None:
        raise RuntimeError("torch unavailable")
    import cv2

    W0, H0 = frame.size
    height = height or H0
    width = width or W0
    steps = int(steps or PROBE_STEPS)
    seeds = seeds or PROBE_SEEDS
    quantile = PROBE_QUANTILE if quantile is None else quantile

    a = anchor_text if anchor_text is not None else globals().get("ANCHOR", "")
    h2, w2 = height // TOKEN_STRIDE, width // TOKEN_STRIDE

    acc, used = None, 0
    for sd in seeds:
        g = torch.Generator(device=_exec_device(pipe)).manual_seed(int(sd))
        z0 = encode_reference(pipe, frame, generator=g).to(torch.float32)
        eps = torch.randn(z0.shape, generator=g, device=z0.device, dtype=torch.float32)

        tQ = _probe_trajectory(pipe, frame, clean, a + instruction, eps, steps,
                               cfg, neg, height, width, sd)
        tR = _probe_trajectory(pipe, frame, clean, a + PROBE_NULL, eps, steps,
                               cfg, neg, height, width, sd)
        n = min(len(tQ), len(tR))
        if n == 0:
            continue
        for k in range(n):
            if tQ[k].shape != tR[k].shape:
                continue
            d = (tQ[k] - tR[k]).abs().mean(dim=-1)[0]      # (seq,) per-token
            acc = d if acc is None else acc + d
        used += 1

    if acc is None or used == 0:
        return None, dict(reject="probe produced no comparable latents")
    if acc.numel() != h2 * w2:
        return None, dict(reject=f"token count {acc.numel()} != {h2}x{w2}={h2*w2}")

    m = acc.reshape(h2, w2).float().cpu().numpy()
    m = m - m.min()
    m = m / (m.max() + 1e-8)
    if PROBE_SMOOTH_TOK >= 3:
        k = PROBE_SMOOTH_TOK | 1
        m = cv2.GaussianBlur(m, (k, k), 0)

    thr = float(np.quantile(m, quantile))
    binm = (m > thr).astype(np.float32)
    frac = float(binm.mean())
    info = dict(frac=round(frac, 4), thr=round(thr, 4), seeds=used,
                heat=m, reject="")

    if save_heatmap:
        hm = (255 * m / (m.max() + 1e-8)).astype(np.uint8)
        hm = cv2.applyColorMap(cv2.resize(hm, (width, height),
                                          interpolation=cv2.INTER_NEAREST),
                               cv2.COLORMAP_INFERNO)
        Image.fromarray(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)).save(save_heatmap)
        info["heatmap_path"] = save_heatmap

    if frac < PROBE_MIN_FRAC:
        info["reject"] = f"degenerate: only {100*frac:.1f}% of tokens above threshold"
        return None, info
    if frac > PROBE_MAX_FRAC:
        info["reject"] = f"not constraining: {100*frac:.1f}% of tokens editable"
        return None, info

    px = cv2.resize(binm, (width, height), interpolation=cv2.INTER_NEAREST)
    _log(f"instruction-derived mask: {100*frac:.1f}% of tokens editable "
         f"(quantile {quantile}, {used} seed(s), {steps} probe steps)")
    return (px > 0.5).astype(np.uint8), info


def mask_from_bbox(bbox, H, W, pad_frac=0.10):
    """Editable region from a tracked object bbox (the step>=2 case)."""
    m = np.zeros((H, W), np.uint8)
    if bbox is None:
        return m + 1
    x0, y0, x1, y1 = [int(v) for v in bbox]
    px, py = int(pad_frac * W), int(pad_frac * H)
    m[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)] = 1
    return m
    
def diagnose(pipe, image, steps=8):
    W, H = image.size
    print("=" * 70)
    print("LATENT LAYOUT DIAGNOSTIC  (v11)")
    print("=" * 70)
    print(f"image            : {W}x{H}  (H%16={H%16}, W%16={W%16})")
    print(f"expected latent  : {H//VAE_DOWN}x{W//VAE_DOWN}")
    print(f"expected tokens  : {H//TOKEN_STRIDE}x{W//TOKEN_STRIDE} = "
          f"{(H//TOKEN_STRIDE)*(W//TOKEN_STRIDE)}")
    z = encode_reference(pipe, image)
    print(f"packed reference : {tuple(z.shape)}")
    exp = (H // TOKEN_STRIDE) * (W // TOKEN_STRIDE)
    print(f"token count match: {z.shape[1] == exp}  ({z.shape[1]} vs {exp})")

    seen = {}
    def probe(p, step, t, kw):
        lt = kw.get("latents")
        if lt is not None and "shape" not in seen:
            seen["shape"] = tuple(lt.shape)
        sg = getattr(p.scheduler, "sigmas", None)
        if sg is not None and len(sg):
            seen.setdefault("n_sigmas", len(sg))
            seen["last_sigma"] = float(sg[min(int(step)+1, len(sg)-1)])
            seen.setdefault("first_sigma", float(sg[min(int(step)+1, len(sg)-1)]))
        seen["fired"] = seen.get("fired", 0) + 1
        return kw

    g = torch.Generator(device=_exec_device(pipe)).manual_seed(0)
    with torch.inference_mode():
        pipe(image=[image, image], prompt="no change", negative_prompt=" ",
             true_cfg_scale=1.0, num_inference_steps=steps, guidance_scale=1.0,
             height=H, width=W, generator=g,
             callback_on_step_end=probe,
             callback_on_step_end_tensor_inputs=["latents"])

    print(f"callback fired   : {seen.get('fired', 0)} times (expected {steps})")
    print(f"live latents     : {seen.get('shape')}")
    print(f"sigmas visible   : {seen.get('n_sigmas')} entries "
          f"(schedule readable from inside the callback = the v11 fix)")
    print(f"sigma first/last : {seen.get('first_sigma')} -> {seen.get('last_sigma')}")

    ok_tokens = seen.get("shape", (0, 0))[1] == z.shape[1]
    ok_sigmas = (seen.get("n_sigmas") or 0) > 0
    ok_end    = (seen.get("last_sigma") is not None and seen["last_sigma"] <= 0.05)
    print(f"tokens ALIGNED   : {ok_tokens}")
    print(f"sigmas LIVE      : {ok_sigmas}")
    print(f"ends at sigma~0  : {ok_end}")
    return bool(ok_tokens and ok_sigmas)