"""Generate the coding-agent scenario clips (see core/agent_scenarios).

    py -m great_sage.voice.make_agent_clips            # only missing clips
    py -m great_sage.voice.make_agent_clips --force    # remake all of them

Run once, and again after changing a line in CATALOGUE. Each clip is the
real recorded opener (koku 告 / kai 解), a short pause, then XTTS speaking
the Japanese line in the Great Sage voice - one file, so it plays with no
seam. Needs the GPU only while it runs; the app never loads XTTS for them.
"""
import argparse
from math import gcd
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from great_sage.config import settings
from great_sage.core.agent_scenarios import CATALOGUE, CLIP_DIR

RATE = 24000
# XTTS speaking rate, matched to the real recordings by measurement
# (2026-10-02): recorded lines speak a median 7.50 mora/s; XTTS gave 6.12
# at speed 1.00 (82%, the slowness Master heard), 7.83 at 1.15. 1.12
# interpolates to the recordings' pace.
SPEED = 1.12
ROOT = Path(__file__).resolve().parents[2]
OPENERS = {'koku': ROOT / 'voice_lines' / 'koku.ogg', 'kai': ROOT / 'voice_lines' / 'kai.ogg',
           'kidou': ROOT / 'voice_lines' / 'kidou.ogg'}


def _load(path):
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if rate != RATE:
        g = gcd(RATE, rate)
        data = resample_poly(data, RATE // g, rate // g).astype(np.float32)
    return data


def _silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='remake clips that exist')
    args = parser.parse_args()
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(name, i, opener, ja) for name, (opener, lines) in CATALOGUE.items()
            for i, (ja, _) in enumerate(lines)
            if args.force or not (CLIP_DIR / f'{name}_{i}.wav').exists()]
    if not todo:
        print('All scenario clips exist.')
        return

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    model_dir = Path(getattr(settings, 'JAPANESE_MODEL_DIR', ROOT / '.model-cache' / 'xtts-v2'))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    reference = Path(getattr(settings, 'JAPANESE_REFERENCE_AUDIO_PATH', 'voice_samples/great_sage_japanese.wav'))
    if not reference.is_absolute():
        reference = ROOT / reference
    config = XttsConfig()
    config.load_json(str(model_dir / 'config.json'))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    latents = model.get_conditioning_latents(audio_path=[str(reference)])
    openers = {key: _load(path) for key, path in OPENERS.items()}

    for name, i, opener, ja in todo:
        parts = [_silence(0.15)]
        if ja.startswith('@'):
            # A real recording, used as-is: no opener, no synthesis.
            audio = np.concatenate([_silence(0.15), _load(ROOT / 'voice_lines' / f'{ja[1:]}.ogg'), _silence(0.3)])
            out = CLIP_DIR / f'{name}_{i}.wav'
            sf.write(str(out), audio / max(float(np.max(np.abs(audio))) or 1.0, 1.0), RATE, subtype='PCM_16')
            print(f'{out.name}: {len(audio) / RATE:.1f}s  recorded {ja[1:]}.ogg')
            continue
        if opener:
            parts.append(openers[opener])
        if ja:  # An empty line is the recorded opener alone (the greeting).
            with torch.inference_mode():
                wav = model.inference(ja, 'ja', *latents, temperature=0.6, speed=SPEED,
                                      enable_text_splitting=True)['wav']
            speech = wav.float().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
            if opener:
                parts.append(_silence(0.12))
            parts.append(speech)
        parts.append(_silence(0.3))
        audio = np.concatenate(parts)
        peak = float(np.max(np.abs(audio))) or 1.0
        out = CLIP_DIR / f'{name}_{i}.wav'
        sf.write(str(out), audio / max(peak, 1.0), RATE, subtype='PCM_16')
        print(f'{out.name}: {len(audio) / RATE:.1f}s  {ja}')


if __name__ == '__main__':
    os.environ.setdefault('COQUI_TOS_AGREED', '1')
    main()
