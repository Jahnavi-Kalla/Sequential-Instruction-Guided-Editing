#Kontext backend
import gc
import numpy as np
from PIL import Image

KONTEXT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
KONTEXT_GUIDANCE = 2.5
KONTEXT_STEPS    = 28
KONTEXT_USE_NEG  = False
KONTEXT_ANCHOR   = True

_KTX_KEY = "_FLUX_KONTEXT_PIPE"
_KTX_INP_KEY = "_FLUX_KONTEXT_INPAINT_PIPE"


def _free_everything_else():
    for name in ("free_pipe", "free_gen_pipe", "free_vlm", "free_pie_models"):
        fn = globals().get(name)
        if callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"[kontext] {name}() raised ({e}); continuing")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_kontext_pipe():
    import builtins
    if getattr(builtins, _KTX_KEY, None) is not None:
        return getattr(builtins, _KTX_KEY)
    import torch
    _free_everything_else()                                            # [K2]
    from diffusers import FluxKontextPipeline
    print(f"[kontext] loading {KONTEXT_MODEL_ID} (bf16) ...")
    try:
        pipe = FluxKontextPipeline.from_pretrained(KONTEXT_MODEL_ID,
                                                   torch_dtype=torch.bfloat16)
    except Exception as e:
        print("[kontext] LOAD FAILED. The usual cause is the gated licence:")
        print("          open https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev")
        print("          accept the licence, then re-run cell00_login.")
        raise
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    builtins.__dict__[_KTX_KEY] = pipe
    if torch.cuda.is_available():
        print(f"[vram] after Kontext load allocated="
              f"{torch.cuda.memory_allocated()/2**30:.2f} GB")
    return pipe


def get_kontext_inpaint():
    import builtins
    if getattr(builtins, _KTX_INP_KEY, None) is not None:
        return getattr(builtins, _KTX_INP_KEY)
    base = get_kontext_pipe()
    from diffusers import FluxKontextInpaintPipeline
    pipe = FluxKontextInpaintPipeline(**{k: getattr(base, k) for k in base.components})
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    builtins.__dict__[_KTX_INP_KEY] = pipe
    print("[kontext] inpaint pipeline built from loaded components (no second load)")
    return pipe


def free_kontext_pipe():
    import builtins
    for k in (_KTX_INP_KEY, _KTX_KEY):
        if getattr(builtins, k, None) is not None:
            builtins.__dict__[k] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[kontext] freed")


class KontextBackend:
    tag = "kontext"
    label = "FLUX.1-Kontext-dev"

    def supports(self, arm):
        return arm in ("composite", "partial", "latent", "spotedit")

    def load(self):
        return get_kontext_pipe()

    def free(self):
        free_kontext_pipe()
    def _prompt(self, instr):
        if KONTEXT_ANCHOR and globals().get("ANCHOR"):
            return f"{ANCHOR}{instr}".strip()
        return instr

    def _common(self, frame):
        import torch
        W, H = frame.size
        return dict(prompt_holder=None, height=H, width=W,
                    generator=torch.Generator(device="cpu").manual_seed(int(SEED)))

    def _finish(self, out, frame):
        out = out.convert("RGB")
        return out if out.size == frame.size else out.resize(frame.size, Image.BICUBIC)
    def edit_full(self, frame, clean, prompt):
        import torch
        W, H = frame.size
        kw = dict(image=frame, prompt=self._prompt(prompt),
                  guidance_scale=KONTEXT_GUIDANCE,
                  num_inference_steps=KONTEXT_STEPS,
                  height=H, width=W,
                  generator=torch.Generator(device="cpu").manual_seed(int(SEED)))
        if KONTEXT_USE_NEG and globals().get("NEG"):
            kw["negative_prompt"] = NEG
        with torch.inference_mode():
            out = get_kontext_pipe()(**kw).images[0]
        return self._finish(out, frame)

    def edit_partial(self, frame, clean, prompt, strength):
        import torch
        W, H = frame.size
        white = Image.new("L", (W, H), 255)
        kw = dict(image=frame, mask_image=white, prompt=self._prompt(prompt),
                  strength=float(strength), guidance_scale=KONTEXT_GUIDANCE,
                  num_inference_steps=KONTEXT_STEPS, height=H, width=W,
                  generator=torch.Generator(device="cpu").manual_seed(int(SEED)))
        try:
            with torch.inference_mode():
                out = get_kontext_inpaint()(**kw).images[0]
            print(f"    [partial/kontext] strength={strength}")
        except Exception as e:
            print(f"    [partial/kontext] inpaint pipeline failed "
                  f"({type(e).__name__}: {e}) -> FULL NOISE fallback, "
                  f"STRENGTH NOT APPLIED this step.")
            return self.edit_full(frame, clean, prompt)
        return self._finish(out, frame)
    def edit_masked(self, frame, clean, prompt, mask_hw):
        import torch
        info = {"mechanism": "FluxKontextInpaintPipeline + derived mask"}
        W, H = frame.size
        m = np.clip(np.asarray(mask_hw, np.float32), 0, 1)
        if m.shape != (H, W):
            import cv2
            m = cv2.resize(m, (W, H))
        mimg = Image.fromarray((m * 255).astype(np.uint8), mode="L")
        kw = dict(image=frame, mask_image=mimg, prompt=self._prompt(prompt),
                  strength=1.0, guidance_scale=KONTEXT_GUIDANCE,
                  num_inference_steps=KONTEXT_STEPS, height=H, width=W,
                  generator=torch.Generator(device="cpu").manual_seed(int(SEED)))
        try:
            with torch.inference_mode():
                out = get_kontext_inpaint()(**kw).images[0]
        except Exception as e:
            print(f"    [latent/kontext] masked edit FAILED "
                  f"({type(e).__name__}: {e}) -> re-running UNMASKED and marking it.")
            info["unconstrained"] = 1
            info["error"] = f"{type(e).__name__}: {e}"
            return self.edit_full(frame, clean, prompt), info
        return self._finish(out, frame), info


KONTEXT = KontextBackend()