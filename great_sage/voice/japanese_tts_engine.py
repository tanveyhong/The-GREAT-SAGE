"""Local Japanese anime voice cloning, with translated speech and HUD playback.

English stays the written language: captions carry the English each clip
was translated from, while the audio is Japanese. Translation goes through
a ModelProvider so the translating model can be swapped like any other.
"""
from collections import OrderedDict
import contextlib
import gc
import logging
import queue
import threading
import time
import uuid
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

# Streaming (see _speak_items_streamed). XTTS yields audio every this many
# GPT tokens; smaller starts sooner but sends more, shorter chunks.
STREAM_CHUNK_TOKENS = 20
PIECE_GAP = np.zeros(int(0.1 * RATE), dtype=np.float32)   # between XTTS pieces
CACHED_CHUNK = int(0.5 * RATE)          # cached audio is re-sent in this size
STREAM_TAIL_SECONDS = 0.3               # silence that lets the reverb ring out
# How far synthesis may run ahead of playback. Enough to ride out a slow
# chunk or a translation, little enough that the GPU works at a steady pace.
PACE_AHEAD_SECONDS = 3.0


# Agents repeat themselves - "Now running the tests.", the same openers -
# so recent lines skip translation (and, when short, synthesis) entirely.
TRANSLATION_CACHE_SIZE = 300
AUDIO_CACHE_SIZE = 64          # ~30MB at most: a few seconds of float32 each
AUDIO_CACHE_MAX_CHARS = 120    # Japanese characters; longer lines rarely repeat


class _LRU(OrderedDict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def get_recent(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def keep(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


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
                 summary_threshold=700, summary_sentences=3, autocast=True, streaming=True):
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
        self.streaming = streaming
        self.voice_lines = voice_lines or []
        self._disabled_patterns = set(disabled_voice_line_patterns or ())
        self._single_shot = True
        self._sink = LocalSpeakerSink()
        self._stop_requested = False
        self.speaking = False
        self.fx = VoiceFX()
        self._translations = _LRU(TRANSLATION_CACHE_SIZE)  # English -> Japanese
        self._audio = _LRU(AUDIO_CACHE_SIZE)               # Japanese -> (samples, rate)
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
        cached = self._translations.get_recent(english)
        if cached:
            return cached
        system = self._translate_prompt
        for attempt in (1, 2):
            translated = jt.fix_japanese(self._ask(system, english), self.glossary)
            problem = jt.problem(english, translated)
            if problem is None:
                self._translations.keep(english, translated)
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
        if self.streaming and getattr(self._sink, 'supports_streaming', False):
            return self._speak_items_streamed(items)
        return self._speak_items_clips(items)

    def _translate_into(self, items, out, failure, put):
        """The translation stage, shared by both ways of speaking: runs ahead
        on its own thread, passing recorded clips through in order."""
        try:
            for kind, payload, *rest in items:
                if self._stop_requested or failure:
                    break
                if kind == 'audio':
                    put(out, ('file', payload, rest[0]))
                else:
                    japanese = self.translate(payload)
                    if japanese:
                        log.info('Japanese for %r: %s', payload[:60], japanese[:200])
                        put(out, ('text', japanese, payload))
        except BaseException as exc:
            failure.append(exc)
        finally:
            put(out, None)

    def _stream_piece(self, piece):
        """XTTS audio for one piece (<= the 71-char Japanese limit), chunk by
        chunk as it is generated."""
        precision = (torch.autocast('cuda', dtype=torch.float16)
                     if self.autocast else contextlib.nullcontext())
        with torch.inference_mode(), precision:
            for chunk in self._model.inference_stream(
                    piece, 'ja', *self._conditioning,
                    stream_chunk_size=STREAM_CHUNK_TOKENS, temperature=0.65,
                    enable_text_splitting=False):
                if self._stop_requested:
                    return
                yield chunk.float().cpu().numpy() if torch.is_tensor(chunk) \
                    else np.asarray(chunk, dtype=np.float32)

    def _stream_send(self, samples, stream_id, text, final):
        """One chunk to the sink; if the window closes under it, send it to
        the window the voice route moves to and carry on there."""
        sink = self._sink
        try:
            return sink.stream_chunk(samples, RATE, stream_id, text=text, final=final)
        except Exception as exc:
            deadline = time.monotonic() + 3
            while self._sink is sink and time.monotonic() < deadline and not self._stop_requested:
                time.sleep(0.1)
            if self._sink is sink or self._stop_requested:
                raise VoiceError(f'Audio streaming failed: {exc}') from exc
        log.info('Voice moved to another window; continuing the stream there')
        return self._sink.stream_chunk(samples, RATE, stream_id, text=text, final=final)

    def _speak_items_streamed(self, items):
        """Translate ahead on one thread; synthesize with XTTS streaming here,
        sending each chunk as it is made. The page schedules the chunks back
        to back, so a reply is one seamless stream: no clip boundaries to
        clip, and the first sound comes as soon as the first chunk exists."""
        translated = queue.Queue()
        failure = []

        def put(target, item):
            target.put(item)

        worker = threading.Thread(target=self._translate_into, name='japanese-translate',
                                  args=(items, translated, failure, put), daemon=True)
        worker.start()
        stream_id = uuid.uuid4().hex
        began = time.monotonic()
        state = {'sent': 0.0, 'first': None, 'caption': None}

        def send(samples, final=False):
            # The first chunk of each sentence carries its English caption.
            text, state['caption'] = state['caption'], None
            self._stream_send(self.fx.apply(samples, RATE), stream_id, text, final)
            if state['first'] is None:
                state['first'] = time.monotonic()
                log.info('Japanese voice: first audio %.1fs after the reply arrived',
                         state['first'] - began)
            state['sent'] += len(samples) / RATE
            # Pace the GPU: once PACE_AHEAD_SECONDS of audio is queued in the
            # page, wait for playback to catch up before generating more.
            # Flat out, XTTS ran far faster than realtime and then idled - a
            # tall spike in GPU load; paced, the same work is a low plateau.
            while not final and not self._stop_requested:
                ahead = state['sent'] - (time.monotonic() - state['first'])
                if ahead <= PACE_AHEAD_SECONDS:
                    break
                time.sleep(min(ahead - PACE_AHEAD_SECONDS, 0.25))

        def send_held(held):
            for path, words in held:
                state['caption'] = words
                send(_clip_samples(path))

        held = []
        self.speaking = True  # Progress cues stay quiet while this is set.
        try:
            while not self._stop_requested:
                waited = time.monotonic()
                item = translated.get()
                waited = time.monotonic() - waited
                if state['first'] is not None and waited > 0.4 and item is not None:
                    log.info('Japanese voice waited %.1fs for the next translation', waited)
                if item is None or failure:
                    break
                kind, payload, caption = item
                if kind == 'file':
                    # Held, so it flows straight into the sentence after it
                    # instead of playing alone into silence.
                    held.append((payload, caption))
                    continue
                # One caption for the held clips and this sentence, shown as
                # the first of them starts.
                state['caption'] = ' '.join([words for _, words in held] + [caption])
                for path, _ in held:
                    send(_clip_samples(path))
                held = []
                hit = self._audio.get_recent(payload)
                if hit is not None:
                    log.info('Japanese voice reused cached audio for %r', caption[:50])
                    for start in range(0, len(hit[0]), CACHED_CHUNK):
                        send(hit[0][start:start + CACHED_CHUNK])
                    continue
                made = []
                for n, piece in enumerate(jt.split_japanese(payload)):
                    if self._stop_requested:
                        break
                    if n:
                        made.append(PIECE_GAP)
                        send(PIECE_GAP)
                    for chunk in self._stream_piece(piece):
                        made.append(chunk)
                        send(chunk)
                if made and len(payload) <= AUDIO_CACHE_MAX_CHARS and not self._stop_requested:
                    self._audio.keep(payload, (np.concatenate(made), RATE))
            if not self._stop_requested:
                send_held(held)  # Clips with no speech after them.
                # Silence through the effects lets the reverb ring out, and
                # closes the stream; then wait for the page to finish it.
                send(np.zeros(int(STREAM_TAIL_SECONDS * RATE), dtype=np.float32), final=True)
                playing = state['sent'] - (time.monotonic() - (state['first'] or began))
                self._sink.wait_stream_end(max(0.0, playing))
        except BaseException:
            self._stop_requested = True  # Wind the translator down with us.
            raise
        finally:
            self.speaking = False
            worker.join(timeout=60)
        if failure:
            raise VoiceError(f'Japanese speech failed: {failure[0]}') from failure[0]

    def _speak_items_clips(self, items):
        """Three stages, each on its own thread: translate, synthesize, play.
        Used when the sink cannot stream (local speakers).

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
            self._translate_into(items, translated, failure, put)

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
                    hit = self._audio.get_recent(payload)
                    if hit is not None:
                        samples, rate = hit
                        log.info('Japanese voice reused cached audio for %r', caption[:50])
                    else:
                        samples, rate = self.generate(payload)
                        if len(payload) <= AUDIO_CACHE_MAX_CHARS and not self._stop_requested:
                            self._audio.keep(payload, (samples, rate))
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
