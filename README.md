# Sequentiall-Instruction-Guided-Editing
The core research question is sequential consistency: “Can current instruction-guided image editors be organised into a pipeline that produces a sequence of single-step visual instructions which stay consistent across the sequence, make the changed region explicit, and satisfy the accessi-bility requirements of workers with intellectual disabilities?”

Why this project exists
Workers with intellectual disabilities can carry out multi-step manual tasks (assembly, packaging, etc.) more independently when each step is shown as a still image with a single, clearly marked change against an otherwise unchanged scene. Producing such sequences by photography does not scale. This project investigates whether current, openly available instruction-guided image editors can generate them on demand.

What it does
1. Chains single-step edits from one starting scene and one short instruction per step.
2. Compares two rectified-flow editing models — Qwen-Image-Edit-2511 and FLUX.1 Kontext — under a shared pipeline.
3. Compares three editing mechanisms — hard compositing, partial-noise initialisation, and latent masking — against an un-instrumented baseline, holding the rest of the pipeline constant.
4. Preserves the background around a discovered change region and marks that region with one consistent accessibility cue.
5. Evaluates without a reference image using a purpose-built, comprehension-oriented metric suite.

Key findings
1. The instrumented pipeline substantially reduces cumulative drift vs. the baseline across the tested sequences.
2. Partial-noise initialisation is the most reliable mechanism.
3. The approach works for adding and modifying objects, but not for relocating them.
4. Standard preservation- and CLIP-based metrics can rate a visibly poor sequence highly, so human-anchored, comprehension-oriented evaluation is needed.