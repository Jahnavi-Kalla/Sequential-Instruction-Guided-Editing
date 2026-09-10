#Qwen backend
import numpy as np
from PIL import Image


class QwenBackend:
    tag = "qwen"
    label = "Qwen-Image-Edit-2511"

    def supports(self, arm):
        return arm in ("composite", "partial", "latent", "spotedit")
    def load(self):
        return get_pipe()

    def free(self):
        try:
            free_pipe()
        except Exception as e:
            print(f"[qwen] free_pipe() raised ({e})")
    def edit_full(self, frame, clean, prompt):
        out = edit_once(get_pipe(), frame, clean, prompt)
        return out if out.size == frame.size else out.resize(frame.size, Image.BICUBIC)
    def edit_partial(self, frame, clean, prompt, strength):
        return partial_edit(frame, clean, prompt, strength)

    def edit_masked(self, frame, clean, prompt, mask_hw):
        try:
            out = masked_edit(get_pipe(), frame, clean, prompt, mask=mask_hw,
                              cfg=CFG, steps=STEPS, seed=SEED, neg=NEG)
            if isinstance(out, (tuple, list)):
                out = out[0]
        except Exception as e:
            print(f"    [qwen/latent] masked_edit FAILED ({type(e).__name__}: {e}) "
                  f"-> re-running this step UNMASKED and marking it.")
            info["unconstrained"] = 1
            info["error"] = f"{type(e).__name__}: {e}"
            out = self.edit_full(frame, clean, prompt)
        return (out if out.size == frame.size else out.resize(frame.size, Image.BICUBIC)), info

STRENGTH = 0.85


def _sigma_schedule(pipe, steps, frame):
    import torch
    try:
        sch = pipe.scheduler
        W, H = frame.size
        vae_sf = 8
        try:
            vae_sf = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        except Exception:
            pass
        seq = max(1, (H // vae_sf // 2) * (W // vae_sf // 2))
        kw = {}
        cfg = getattr(sch, "config", None)
        if cfg is not None and getattr(cfg, "use_dynamic_shifting", False):
            try:
                from diffusers.pipelines.flux.pipeline_flux import calculate_shift
                kw["mu"] = calculate_shift(
                    seq,
                    getattr(cfg, "base_image_seq_len", 256),
                    getattr(cfg, "max_image_seq_len", 4096),
                    getattr(cfg, "base_shift", 0.5),
                    getattr(cfg, "max_shift", 1.15))
            except Exception:
                b, m = 0.5, 1.15
                bl, ml = 256, 4096
                kw["mu"] = b + (m - b) * (seq - bl) / max(1, (ml - bl))
        import copy
        s2 = copy.deepcopy(sch)
        s2.set_timesteps(int(steps), device="cpu", **kw)
        sig = s2.sigmas.detach().cpu().numpy().astype(float)
        return sig
    except Exception as e:
        print(f"    [partial] could not read the sigma schedule ({e}); "
              f"THIS STEP WILL RUN AT FULL NOISE.")
        return None


def partial_edit(frame, clean, prompt, strength, neg=None, steps=None,
                 cfg=None, seed=None):
    import torch
    pipe = get_pipe()
    neg = NEG if neg is None else neg
    steps = STEPS if steps is None else steps
    cfg = CFG if cfg is None else cfg
    seed = SEED if seed is None else seed
    W, H = frame.size
    a = globals().get("ANCHOR", "")
    dev = _exec_device(pipe)
    g = torch.Generator(device=dev).manual_seed(int(seed))

    call = dict(image=[frame, clean], prompt=a + prompt, negative_prompt=neg,
                true_cfg_scale=cfg, guidance_scale=1.0, num_images_per_prompt=1,
                height=H, width=W, generator=g)

    sig = _sigma_schedule(pipe, steps, frame)
    used = None
    if sig is not None and 0.05 < strength < 0.999:
        below = np.where(sig <= strength)[0]
        idx = int(below[0]) if below.size else 0
        tail = sig[idx:]
        if len(tail) >= 2:
            s0 = float(tail[0])
            z_ref = encode_reference(pipe, frame, generator=g).to(torch.float32)
            eps = torch.randn(z_ref.shape, generator=g,
                              device=z_ref.device, dtype=torch.float32)
            start = (1.0 - s0) * z_ref + s0 * eps        # exact flow interpolant
            ldtype = getattr(getattr(pipe, "transformer", None), "dtype", None) or z_ref.dtype
            call["sigmas"] = [float(x) for x in (tail[:-1] if tail[-1] == 0 else tail)]
            call["latents"] = start.to(dtype=ldtype)
            used = f"start_sigma={s0:.3f}, {len(call['sigmas'])} steps (strength={strength})"
    if "sigmas" not in call:
        call["num_inference_steps"] = int(steps)

    try:
        with torch.inference_mode():
            out = pipe(**call).images[0]
        print(f"    [partial] {used or f'FULL NOISE ({steps} steps) -- strength NOT applied'}")
    except TypeError as e:
        print(f"    [partial] pipe rejected the custom schedule ({e}). "
              f"FALLING BACK to full noise -- STRENGTH WAS NOT APPLIED.")
        for k in ("sigmas", "latents"):
            call.pop(k, None)
        call["num_inference_steps"] = int(steps)
        with torch.inference_mode():
            out = pipe(**call).images[0]
    return out.resize(frame.size) if out.size != frame.size else out


QWEN = QwenBackend()