"""The Coding Agent Companion's voice: recorded clips only, never synthesis.

APP_MODE = "companion" builds this instead of the XTTS engine. It imports
no torch, no TTS library and no model - the whole point: Great Sage then
runs in ~625MB (mostly the HUD window) instead of ~2.5GB, and plays pre-made clips for what
Claude and Codex are doing (core/agent_scenarios) plus the recorded
voice lines (告, 解, ...) wherever a reply contains their phrase.

The server plays scenario clips itself; this object supplies the sink,
the effects chain and the VoiceOutput surface the server expects.
"""
import logging
import os
from typing import List

from great_sage.voice.audio_fx import VoiceFX
from great_sage.voice.base import VoiceError
from great_sage.voice.sinks import LocalSpeakerSink
from great_sage.voice.voice_lines import label_from_pattern, split_voice_lines

log = logging.getLogger(__name__)


class ClipVoiceOutput:
    # Read by the server: relays play scenario clips, synthesize nothing.
    translate_enabled = False
    companion = True

    def __init__(self, voice_lines=None, disabled_voice_line_patterns=None):
        self.voice_lines = voice_lines or []
        self._disabled_patterns = set(disabled_voice_line_patterns or ())
        self._sink = LocalSpeakerSink()
        self._stop_requested = False
        self.speaking = False
        self.fx = VoiceFX()
        self.api_tts = None

    # --- sink and effects -------------------------------------------------
    def set_sink(self, sink) -> None:
        self._sink = sink

    @property
    def current_sink(self):
        return self._sink

    def set_fx(self, **kwargs) -> None:
        self.fx.update(**kwargs)

    def set_reference_audio(self, reference_audio_path) -> None:
        raise VoiceError('The Companion plays recorded clips; it has no voice to clone.')

    # --- voice lines (the HUD's Voice Lines panel) ----------------------
    def set_voice_line_enabled(self, pattern_str: str, enabled: bool) -> None:
        if enabled:
            self._disabled_patterns.discard(pattern_str)
        else:
            self._disabled_patterns.add(pattern_str)

    def list_voice_lines(self) -> List[dict]:
        return [{"pattern": pattern.pattern, "label": label_from_pattern(pattern.pattern),
                 "enabled": pattern.pattern not in self._disabled_patterns,
                 "file": os.path.basename(path)}
                for pattern, path in self.voice_lines]

    def _active_voice_lines(self):
        return [(p, path) for p, path in self.voice_lines if p.pattern not in self._disabled_patterns]

    # --- speaking: recorded clips only ------------------------------------
    def play_file(self, path, caption=None) -> None:
        """One recorded clip through the effects chain, to the sink."""
        import soundfile as sf
        samples, rate = sf.read(str(path), dtype='float32', always_2d=False)
        if getattr(samples, 'ndim', 1) > 1:
            samples = samples.mean(axis=1)
        self.fx.begin_utterance()
        setattr(self._sink, 'pending_text', caption)
        self._sink.play(self.fx.apply(samples, rate), rate)

    def speak(self, text: str) -> None:
        """Great Sage's own replies: play the recorded line for each phrase
        it contains (Notice. Answer. ...); the rest stays as text."""
        self._stop_requested = False
        self.speaking = True
        try:
            for kind, payload, spoken in split_voice_lines(text or '', self._active_voice_lines()):
                if self._stop_requested:
                    break
                if kind == 'audio':
                    self.play_file(payload, spoken)
        finally:
            self.speaking = False

    def speak_stream(self, deltas, timer=None) -> None:
        self.speak(''.join(deltas))

    def stop(self) -> None:
        self._stop_requested = True
        try:
            self._sink.stop()
        except Exception:
            pass
