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
from pathlib import Path

import numpy as np
import torch

from great_sage.models.base import ModelProviderError
from great_sage.voice.f5_tts_engine import F5TTSVoiceOutput
from great_sage.voice.audio_fx import VoiceFX
from great_sage.voice.sinks import LocalSpeakerSink
from great_sage.voice.base import VoiceError
from great_sage.voice.voice_lines import split_voice_lines
from great_sage.voice import japanese_translation as jt

log = logging.getLogger(__name__)


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
        self.fx = VoiceFX()

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
        return (np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)), 24000

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
            try:
                while True:
                    item = translated.get()
                    if item is None or self._stop_requested or failure:
                        break
                    kind, payload, caption = item
                    if kind == 'text':
                        item = ('wav', self.generate(payload), caption)
                    if not self._stop_requested:
                        put(ready, item)
            except BaseException as exc:
                failure.append(exc)
            finally:
                put(ready, None)

        workers = [threading.Thread(target=t, name=n, daemon=True) for t, n in (
            (translate_all, 'japanese-translate'), (synthesize_all, 'japanese-synth'))]
        for worker in workers:
            worker.start()
        try:
            while True:
                item = ready.get()
                if item is None or self._stop_requested:
                    break
                kind, payload, caption = item
                # The caption is the English this clip came from.
                setattr(self._sink, 'pending_text', caption)
                if kind == 'file':
                    log.info('Playing Japanese voice line: %s', Path(payload).name)
                    self._play_audio_file(payload)
                else:
                    self._play(*payload)
        except BaseException:
            self._stop_requested = True  # Wind the workers down with us.
            raise
        finally:
            # A stopped worker finishes at most its current translation or
            # XTTS piece; waiting keeps two syntheses off the GPU at once.
            if self._stop_requested:
                put(translated, None)
            for worker in workers:
                worker.join(timeout=60)
        if failure:
            raise VoiceError(f'Japanese speech failed: {failure[0]}') from failure[0]

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
        starts_with_clip = any(pattern.match(text) for pattern, _ in self._active_voice_lines())
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
