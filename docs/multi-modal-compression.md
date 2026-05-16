# Multi-Modal Compression (Mode 1 extension)

Status: **design doc**. Not implemented. Author: Jamie Obala. Last updated 2026-05.

## Why this exists

The current Liquid Memory proxy compresses text-only prompts. Modern
cloud LLMs (GPT-4o, Gemini 1.5/2.0, Claude 3.5/3.7) accept images and
audio as input, and those inputs are priced in tokens at rates that
make text look cheap:

| Input | Tokens per unit (approx.) | Cost per unit (gpt-4o pricing) |
|---|---|---|
| 1K English chars | ~250 tokens | $0.0006 |
| 1 image (1024x1024) | ~1500-2000 tokens | $0.004-$0.005 |
| 1 minute of audio (16kHz) | ~750-1000 tokens | $0.002-$0.003 |

A single 50-image PDF can dwarf a 50K-token text document on the bill.
Customers doing OCR pipelines, slide-deck analysis, technical-drawing
review, or contact-center transcript analysis have multi-modal inputs
that the current proxy can't touch.

Goal of this design: a feasible path to a 3-5x compression on image
and audio inputs, sitting in the same `liquid-memory start` proxy
the text path already uses, without changing the customer's request
shape.

## What "compression" means for each modality

Not all of these are the same problem.

### Images

Cloud LLMs internally tile every input image into ~14x14 pixel patches
and feed those patches through a vision encoder. A 1024x1024 image at
the default detail becomes ~1500 patch tokens. The user pays for every
patch token, including the (vast majority of) patches that are
background, whitespace, low-information texture.

The compression opportunity is to drop patches the model would have
ignored anyway. Approaches in order of feasibility:

1. **Saliency-based patch pruning**. Pre-process the image with a
   small open-source vision model (CLIP-ViT, DINOv2). Score each
   patch by attention magnitude. Drop bottom 60-70% by score. Send
   only the surviving patches to the cloud LLM via the URL-encoded
   sparse-image format some providers accept.
   - Pro: no retraining, off-the-shelf models, runs on the same GPU
     as the text extractor.
   - Con: only some providers accept sparse-patch input. OpenAI doesn't
     today; Gemini partially does via region-of-interest hints.

2. **Cropping + downscaling**. Detect the bounding box of "interesting"
   regions, crop, downscale the background. Lossy but format-agnostic
   (it's just a smaller standard image at the API).
   - Pro: works with every provider that takes images.
   - Con: degrades whole-image understanding tasks ("describe the
     scene"). Best for OCR / chart-reading / form-extraction
     workloads where one region carries all the signal.

3. **OCR + send text only**. For images that are mostly text
   (screenshots, scanned documents), run a local OCR model (Tesseract
   or Donut), strip the image, send only the extracted text to the
   cloud LLM.
   - Pro: enormous compression (image -> ~200 text tokens).
   - Con: only works for text-bearing images, but that's the bulk of
     enterprise use cases (contracts, receipts, slide decks).

Path 3 has the highest ROI for the legal/healthcare/finance ICP we
already target. It's also the closest in spirit to the existing text
extraction pipeline (run a local model, extract structured content,
forward only the extracted content). Recommend shipping Path 3 first,
then Path 2 as a fallback for "image-as-scene" requests, then Path 1
only if a provider adds proper sparse-patch input.

### Audio

Cloud LLMs accept audio in one of two ways: raw waveform input
(GPT-4o-audio, Gemini) priced at ~750-1000 tokens per minute, or
pre-transcribed text input priced at the standard text rate.

Compression here is mostly "transcribe locally, send text." This is
the same pattern as Path 3 above. A local Whisper-large-v3 model runs
~30x faster than realtime on a single L4, so an hour of audio
transcribes in ~2 minutes and the resulting transcript is roughly 10K
text tokens vs. ~50K audio tokens. **5x compression with the existing
infrastructure pattern.**

The complication is that some workloads need the model to hear the
audio directly (tone of voice in a sales-call analysis, accent
detection, emotion classification). For those, raw audio must pass
through. Provide a per-request override flag.

### Video

Skip. Video is multiple orders of magnitude more expensive to process
than image or audio, and the cloud LLM APIs that accept video are
still early. Revisit after the image and audio paths are shipped and
generating revenue.

## Architecture

The proxy already has a clean extractor / synthesizer split. The
multi-modal extension is a new set of extractors, all sitting behind
the same interface as the text extractor, with input-type routing at
the top of the request handler.

```
                   POST /v1/chat/completions
                            |
                            v
                  +-------------------+
                  |  Content router   |
                  +-------------------+
                   /      |       \
                  /       |        \
       text doc /  image  |  audio  \  passthrough
              /           |          \   (small input)
             v            v           v
   +----------------+ +---------+ +---------+
   | text extractor | | OCR /   | | Whisper |
   | (today, Mistral)| | vision  | | local   |
   +----------------+ +---------+ +---------+
              \           |          /
               \          |         /
                v         v        v
              +-------------------+
              | unified fact pack |
              +-------------------+
                       |
                       v
              +-------------------+
              | LiteLLM synthesis |
              | (cloud or local)  |
              +-------------------+
```

All four extractor outputs land in the same JSON fact-pack shape, so
the synthesizer code path is unchanged. Adding a new modality is just
adding a new extractor class that implements `extract(input) -> dict`.

## Stages

### Stage 0: scaffolding (~1 week)

- Refactor `proxy.py` so the current text extractor sits behind an
  `Extractor` ABC.
- Add a content-router that dispatches by MIME type (already in the
  OpenAI chat-completions schema via `image_url` and `input_audio`
  message content blocks).
- Cache layer (`liquid_memory/cache.py`, already implemented per
  item #8) keys by `(content_hash, modality, extractor_id)` instead
  of `(content_hash, extractor_id)`.
- No new ML work. Just plumbing. Ships behind a feature flag, defaults
  off.

### Stage 1: audio path (~2-3 weeks)

- Add `WhisperExtractor` backed by `faster-whisper` (CTranslate2 build,
  10-30x faster than HF Whisper on the same hardware).
- Bench against the cost target: a 30-minute call should transcribe in
  under 90 seconds end-to-end on an L4. If not, optimise or drop to
  base/medium models for non-critical workloads.
- Per-request override: `{ "x-liquid-memory": { "audio": "passthrough" } }`
  custom extension to force raw audio through to the cloud LLM.
- Ship behind `LM_FEATURES=audio` env flag. Beta with 3 pilot
  customers before defaulting on.

### Stage 2: image path - OCR-only first (~2 weeks)

- Add `OCRExtractor` backed by Donut or PaddleOCR. Donut wins on
  layout-aware content (contracts with tables); PaddleOCR wins on
  speed for receipt / form extraction. Probably ship both, switch by
  workload hint or auto-detect.
- For images detected as "non-textual" (scene photos, charts without
  legible text), fall back to passthrough.
- Same per-request override pattern as audio.
- Ship behind `LM_FEATURES=image_ocr`.

### Stage 3: image path - saliency / region cropping (~4-6 weeks)

- Only after Stage 2 has been in production for a month and we have
  customer feedback on what fraction of images are non-textual.
- Implement Path 1 (saliency pruning) or Path 2 (crop + downscale)
  depending on what providers will accept at that point.
- This is where the real ML research is, and it should not block the
  audio + OCR shipping path.

### Stage 4: pricing + billing changes (~1 week)

- Cost meters for image + audio extraction (GPU-seconds per
  modality) so the per-customer cost model stays accurate.
- New per-modality compression-ratio numbers on the website /cost-
  analysis tool (see item #7).
- Update the homepage compression number from "5.7x" to "5.7x text,
  Nx image, Mx audio" once N and M are measured in production.

## Resource estimate

Two engineering months for stages 0-2 (audio + OCR shipped to beta
customers). Four months for stages 0-4 (image saliency path shipped,
billing integrated). Doable by one strong infra engineer.

Hardware: a single L4 (24GB VRAM) handles the text extractor +
Whisper-large + Donut concurrently. No new GPU procurement needed for
the beta rollout; existing pilot-customer extractor boxes are sized
for it.

## What this does NOT do

- Does not let the cloud LLM see images / audio it could not see
  before. We compress what we send, we don't add new content paths.
- Does not handle modality conversions where the OUTPUT is multi-modal
  (image generation, voice synthesis). Output-side compression is a
  different problem, possibly a different product.
- Does not solve the "model can't read this specific chart" failure
  mode. If GPT-4o misreads a chart in the raw image, GPT-4o will
  misread it in our cropped image too. We don't improve the model's
  vision; we improve the cost of using that vision.

## Open questions

1. Provider compatibility: which providers accept sparse-patch image
   input today, vs. requiring a full image? List needs to be re-
   confirmed at Stage 3 kickoff because vendor APIs change quarterly.
2. Latency: OCR + image extraction adds 100-500ms to request TTFB.
   For chat use cases this is fine; for real-time voice agents it is
   not. Need a sync / async mode toggle.
3. Privacy: image OCR creates a textual representation of the image
   inside our extractor. Customers in regulated industries may need
   the text-form of their images NEVER to land on disk. The cache
   from item #8 needs an opt-out for image content.
4. Legal: some images contain PII (faces, license plates). Do we run a
   redaction step before extraction? At what point in the pipeline?

These get answered during Stage 0 + Stage 1 implementation. None
block this design from being approved as a roadmap commitment.
