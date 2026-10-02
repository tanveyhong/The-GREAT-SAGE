"""Local Japanese anime voice cloning, with translated speech and HUD playback.

English stays the written language: captions carry the English each clip
was translated from, while the audio is Japanese. Translation goes through
a ModelProvider so the translating model can be swapped like any other.
"""
import contextlib
import gc
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from great_sage.models.base import ModelProviderError
from great_sage.voice.f5_tts_engine import F5TTSVoiceOutput, _pad_tail
from great_sage.voice.audio_fx import VoiceFX
from great_sage.voice.sinks import LocalSpeakerSink
from great_sage.voice.base import VoiceError
from great_sage.voice.voice_lines import split_voice_lines
from great_sage.voice import japanese_translation as jt

log = logging.getLogger(__name__)

RATE = 24000
# Silence before every synthesized clip. XTTS starts speaking within 0-19ms
# of a clip's start (measured), and the HUD starting a new audio element
# swallows that much - heard as the front of each sentence clipped.
LEAD_IN = int(0.15 * RATE)


def _clip_samples(path):
    """A recorded voice line as mono float32 at RATE, to join to speech."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    data, rate = sf.read(path, dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if rate != RATE:
        g = gcd(RATE, rate)
        data = resample_poly(data, RATE // g, rate // g).astype(np.float32)
    return data


def _log_memory(label):
    try:
        import psutil
        ram = psutil.Process().memory_info().rss / 2**20
    except Exception:
        ram = float('nan')
    gpu = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    log.info('Japanese voice memory %s: RAM %.0f MB, GPU %.0f MB', label, ram, gpu)


class JapaneseVoiceOutput(F5TTSVoiceOutput):
    def __init__(self, reference_audio_path, model_dir, translator, voice_lines=None,
                 disabled_voice_line_patterns=None, glossary_path=None,
                 summary_threshold=700, summary_sentences=3, autocast=True):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        config = XttsConfig()
        config.load_json(str(Path(model_dir) / 'config.json'))
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
        self._cuda = torch.cuda.is_available()
        self._model.to('cuda' if self._cuda else 'cpu')
        # The checkpoint is read into RAM before it moves to the GPU; hand
        # back what that copy left behind.
        gc.collect()
        if self._cuda:
            torch.cuda.empty_cache()
        _log_memory('after loading XTTS')
        self._memory_logged = False
        self._reference_audio_path = str(reference_audio_path)
        self._reference_text = ''
        self._ref_cache = None
        self._rebuild_reference()
        self.translator = translator
        self.glossary = jt.load_glossary(glossary_path)
        self._translate_prompt = jt.translate_prompt(self.glossary)
        self.summary_threshold = summary_threshold
        self.summary_sentences = summary_sentences
        self.autocast = autocast and self._cuda
        self.voice_lines = voice_lines or []
        self._disabled_patterns = set(disabled_voice_line_patterns or ())
        self._single_shot = True
        self._sink = LocalSpeakerSink()
        self._stop_requested = False
        self.speaking = False
        self.fx = VoiceFX()
        self.warm_up()

    def warm_up(self):
        """One tiny synthesis at startup. The first XTTS run in a process
        took 9s for a 4.7s clip (kernel and cache set-up), paid by
        whichever reply came first."""
        try:
            self.generate('はい。')
        except Exception:
            log.exception('XTTS warm-up failed; the first reply will be slower')

    def warm_translator(self):
        """Master just sent an agent a prompt, so a reply is coming: load the
        translation model now, in the background, if the provider can."""
        warm = getattr(self.translator, 'warm', None)
        if warm:
            threading.Thread(target=warm, name='translator-warm', daemon=True).start()

    def _rebuild_reference(self):
        self._conditioning = self._model.get_conditioning_latents(
            audio_path=[self._reference_audio_path])

    def set_reference_audio(self, reference_audio_path):
        raise VoiceError('Japanese voice uses the bundled Japanese Great Sage reference.')

    def _ask(self, system, text):
        try:
            return self.translator.send_message([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': text}]).strip()
        except ModelProviderError as exc:
            raise VoiceError(f'Translation model unavailable: {exc}') from exc

    def translate(self, text):
        """Japanese for one unit of English, or None if no attempt was usable."""
        english = jt.prepare_english(text, self.glossary)
        if not english:
            return None
        system = self._translate_prompt
        for attempt in (1, 2):
            translated = jt.fix_japanese(self._ask(system, english), self.glossary)
            problem = jt.problem(english, translated)
            if problem is None:
                return translated
            log.warning('Japanese translation rejected (%s), attempt %d: %r',
                        problem, attempt, translated[:120])
            system = self._translate_prompt + (
                '\nYour previous attempt was rejected: ' + problem +
                '. Return a faithful, complete Japanese translation and nothing else.')
        return None  # Skip the unit: wrong speech is worse than a gap.

    def summarize(self, text):
        summary = self._ask(
            jt.SUMMARY_PROMPT.format(sentences=self.summary_sentences), text)
        return summary if summary and len(summary) < len(text) else text

    def generate(self, text):
        samples = []
        for piece in jt.split_japanese(text):
            if self._stop_requested:
                break
            precision = (torch.autocast('cuda', dtype=torch.float16)
                         if self.autocast else contextlib.nullcontext())
            with torch.inference_mode(), precision:
                result = self._model.inference(
                    piece, 'ja', *self._conditioning,
                    enable_text_splitting=True, temperature=0.65)
            wav = result['wav']
            wav = wav.float().cpu().numpy() if torch.is_tensor(wav) else wav
            samples.extend([np.asarray(wav, dtype=np.float32), np.zeros(2400, dtype=np.float32)])
        if not self._memory_logged:
            self._memory_logged = True
            _log_memory('after first synthesis')
        if not samples:
            return np.zeros(1, dtype=np.float32), RATE
        # The same tail F5 gets: room for the last syllable and the reverb
        # to ring out before the next clip starts; and a lead-in, above.
        return _pad_tail(np.concatenate([np.zeros(LEAD_IN, dtype=np.float32)] + samples), RATE), RATE

    def _plan(self, text):
        """Ordered ('audio', path, words) and ('text', english) items. Clip
        triggers are English phrases, so they are matched before translation."""
        items = []
        for kind, payload, spoken in split_voice_lines(text, self._active_voice_lines()):
            if kind == 'audio':
                items.append(('audio', payload, spoken))
            else:
                items.extend(('text', unit) for unit in jt.speech_units(payload))
        return items

    def _speak_items(self, items):
        """Three stages, each on its own thread: translate, synthesize, play.

        Translation runs ahead of synthesis, and synthesis (about twice
        realtime) ahead of playback, so speech starts after the first short
        unit and later units are ready before the one playing ends."""
        translated = queue.Queue()
        ready = queue.Queue(maxsize=2)
        failure = []

        def put(target, item):
            # Never block for ever on a stage that has stopped reading.
            while True:
                try:
                    return target.put(item, timeout=0.2)
                except queue.Full:
                    if self._stop_requested:
                        return

        def translate_all():
            try:
                for kind, payload, *rest in items:
                    if self._stop_requested or failure:
                        break
                    if kind == 'audio':
                        put(translated, ('file', payload, rest[0]))
                    else:
                        japanese = self.translate(payload)
                        if japanese:
                            log.info('Japanese for %r: %s', payload[:60], japanese[:200])
                            put(translated, ('text', japanese, payload))
            except BaseException as exc:
                failure.append(exc)
            finally:
                put(translated, None)

        def synthesize_all():
            # A recorded clip waits for the speech after it and goes out
            # joined to it as ONE clip. Sent alone, "Notice." played at once
            # and then left seconds of silence while the first sentence was
            # translated, and every extra clip is another seam to clip.
            held = []
            try:
                while True:
                    item = translated.get()
                    if item is None or self._stop_requested or failure:
                        break
                    kind, payload, caption = item
                    if kind == 'file':
                        held.append((payload, caption))
                        continue
                    samples, rate = self.generate(payload)
                    if held:
                        try:
                            lead = [_clip_samples(path) for path, _ in held]
                        except Exception:
                            log.exception('Could not join voice lines; playing them alone')
                            for path, words in held:
                                put(ready, ('file', path, words))
                        else:
                            log.info('Japanese voice line(s) %s joined to the next sentence',
                                     ', '.join(Path(path).name for path, _ in held))
                            samples = np.concatenate(lead + [samples])
                            caption = ' '.join([words for _, words in held] + [caption])
                        held = []
                    if not self._stop_requested:
                        put(ready, ('wav', (samples, rate), caption))
                for path, words in held:  # Clips with no speech after them.
                    if not self._stop_requested:
                        put(ready, ('file', path, words))
            except BaseException as exc:
                failure.append(exc)
            finally:
                put(ready, None)

        workers = [threading.Thread(target=t, name=n, daemon=True) for t, n in (
            (translate_all, 'japanese-translate'), (synthesize_all, 'japanese-synth'))]
        for worker in workers:
            worker.start()
        self.speaking = True  # Progress cues stay quiet while this is set.
        played = 0
        try:
            while True:
                waited = time.monotonic()
                item = ready.get()
                waited = time.monotonic() - waited
                if played and waited > 0.4 and item is not None:
                    # Synthesis fell behind playback: an audible gap.
                    log.info('Japanese voice waited %.1fs for the next clip', waited)
                played += 1
                if item is None or self._stop_requested:
                    break
                kind, payload, caption = item
                # The caption is the English this clip came from.
                setattr(self._sink, 'pending_text', caption)
                if kind == 'file':
                    log.info('Playing Japanese voice line: %s', Path(payload).name)
                self._play_surviving_switch(kind, payload, caption)
        except BaseException:
            self._stop_requested = True  # Wind the workers down with us.
            raise
        finally:
            self.speaking = False
            # A stopped worker finishes at most its current translation or
            # XTTS piece; waiting keeps two syntheses off the GPU at once.
            if self._stop_requested:
                put(translated, None)
            for worker in workers:
                worker.join(timeout=60)
        if failure:
            raise VoiceError(f'Japanese speech failed: {failure[0]}') from failure[0]

    def _play_surviving_switch(self, kind, payload, caption):
        """Play one clip; if its window closes under it, replay it on the
        window the voice route moves to. Switching between the HUD and the
        overlay closes one page while the other takes over, and used to
        abandon the rest of the reply there."""
        sink = self._sink
        try:
            return self._play_audio_file(payload) if kind == 'file' else self._play(*payload)
        except VoiceError:
            deadline = time.monotonic() + 3
            while self._sink is sink and time.monotonic() < deadline and not self._stop_requested:
                time.sleep(0.1)
            if self._sink is sink or self._stop_requested:
                raise
        log.info('Voice moved to another window; replaying the interrupted clip')
        setattr(self._sink, 'pending_text', caption)
        return self._play_audio_file(payload) if kind == 'file' else self._play(*payload)

    def speak(self, text):
        if not text.strip():
            return
        self._stop_requested = False
        self.fx.begin_utterance()
        self._speak_items(self._plan(text))

    def speak_codex(self, text):
        """A reply relayed from Claude or Codex: always opens with the recorded
        Notice cue, and is summarised when it is long."""
        self._stop_requested = False
        self.fx.begin_utterance()
        lines = self._active_voice_lines()
        # The relay adds an opener; one the reply chose itself wins.
        for lead in ('Answer. ', 'Notice. '):
            if text.startswith(lead) and any(p.match(text[len(lead):]) for p, _ in lines):
                text = text[len(lead):]
        starts_with_clip = any(pattern.match(text) for pattern, _ in lines)
        if not starts_with_clip and not text.lstrip().startswith('Notice.'):
            text = 'Notice. ' + text
        items = self._plan(text)
        prose = ' '.join(item[1] for item in items if item[0] == 'text')
        if self.summary_threshold and len(prose) > self.summary_threshold:
            summary = self.summarize(prose)
            log.info('Speaking a %d-character summary of a %d-character reply',
                     len(summary), len(prose))
            # Keep the clips that open the reply, then the summary.
            lead = []
            for item in items:
                if item[0] != 'audio':
                    break
                lead.append(item)
            items = lead + [('text', unit) for unit in jt.speech_units(summary)]
        self._speak_items(items)

    def speak_stream(self, deltas, timer=None):
        # Translate a complete response so Japanese sentence order stays coherent.
        self.speak(''.join(deltas))
