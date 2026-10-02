"""Voice every active session of a coding agent, never driving it.

A relay watches all of an agent's session logs at once - several Claude
or Codex chats can run side by side - and for each one keeps its own read
position, Master's last prompt (for the Answer/Notice opener) and its own
progress-cue timing. What it hears:

    final reply   spoken in full (or summarised), with an opener
    narration     the agent's one-line "now doing X" notes between tool
                  calls, spoken only while fresh and only into a silence
    tool activity recorded progress cues (see progress_cues)

Subclasses only say where the logs are and how to read one record.
"""
from collections import OrderedDict
import json
import logging
from pathlib import Path
import re
import threading
import time

from great_sage.core.agent_scenarios import ScenarioTracker, greeting, note_scenario
from great_sage.core.progress_cues import STALE_SECONDS, CueTracker, record_age
from great_sage.voice.speakable import speakable, cap_for_speech

log = logging.getLogger(__name__)

# How often the log folders are re-listed for new sessions. Known
# sessions are checked for growth on every poll regardless.
RESCAN_SECONDS = 5
# Narration is a one-liner; anything longer was not meant to be heard.
NARRATION_CHARS = 300

_QUESTION_START = re.compile(
    r'^(what|why|how|is|are|was|were|can|could|does|do|did|should|which|who|'
    r'where|when|will|would|shall|has|have)\b', re.IGNORECASE)


def opener(prompt):
    """Great Sage's two openers: 解 "Answer." for a question, else 告 "Notice."."""
    if prompt and ('?' in prompt or _QUESTION_START.match(prompt)):
        return 'Answer.'
    return 'Notice.'


def speech_text(text, limit=6000):
    # Code stays visible in the agent's window; only prose is worth hearing.
    text = re.sub(r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$', '', text)
    text = re.sub(r'(?m)^\s*::[^\n]*$', '', text)
    return cap_for_speech(speakable(text), limit=max(100, min(int(limit), 20000)))


def new_records(path, offset):
    """Yield (record, offset after it) for each complete JSON line past offset."""
    with path.open('rb') as source:
        source.seek(offset)
        while True:
            line = source.readline()
            if not line.endswith(b'\n'):
                return  # End of file, or the writer is mid-record.
            try:
                record = json.loads(line)
            except ValueError:
                record = {}
            yield record, source.tell()


def load_json_config(path, defaults):
    try:
        cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        cfg = {}
    return {**defaults, **cfg}


def _created(stat):
    return getattr(stat, 'st_birthtime', stat.st_ctime)


class Parsed:
    """What one log record means to the relay."""
    __slots__ = ('prompt', 'reply', 'narration', 'events', 'steps', 'folder', 'phases')

    def __init__(self, prompt=None, reply=None, narration=None, events=(), steps=(), folder=None,
                 phases=()):
        self.prompt = prompt          # text Master typed
        self.reply = reply            # (id, text) of a finished turn
        self.narration = narration    # a between-steps note
        self.events = list(events)    # progress_cues tool events
        self.steps = list(steps)      # "Editing x.py" labels for the activity line
        self.folder = folder          # the session's working folder, when the record says
        self.phases = list(phases)    # (call_id, agent_scenarios phase) per tool call


class _Session:
    def __init__(self, offset):
        self.offset = offset
        self.cues = CueTracker()
        self.scenes = ScenarioTracker()
        self.last_prompt = None
        self.label = None  # project folder name, for the activity line


class SessionRelay:
    name = 'agent'

    def __init__(self, speak, config_path, cue=None, narrate=None, activity=None, prepare=None,
                 scenario=None):
        self.prepare = prepare    # a reply is coming: warm the voice up
        self.scenario = scenario  # (name, is_reply) -> plays a pre-made scenario clip
        self.speak = speak        # final reply -> True once handled
        self.cue = cue            # progress cue name -> plays a clip
        self.narrate = narrate    # narration text -> spoken if silent
        self.activity = activity  # dict -> shown on the HUD's activity line
        self.config_path = Path(config_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f'{self.name}-voice-relay', daemon=True)
        self.started = None
        self.last_scan = 0.0
        self.sessions = {}            # path -> _Session
        self.pending = OrderedDict()  # path -> newest unspoken reply
        self.seen = set()

    # --- what a subclass provides ---------------------------------------
    def load_config(self):
        raise NotImplementedError

    def session_files(self, cfg):
        raise NotImplementedError

    def parse(self, record, cfg):
        raise NotImplementedError

    def initial_folder(self, path):
        """The session's folder, read once when it is found, for agents that
        name it only at the top of the log. None to wait for records."""
        return None

    # --- the relay --------------------------------------------------------
    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def clear(self):
        self.pending.clear()

    def scan(self, cfg):
        self.last_scan = time.time()
        for path in self.session_files(cfg):
            if path in self.sessions:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            # Sessions begun after startup are read from the top; older
            # ones from their current end, never their history.
            new = self.started is not None and _created(stat) >= self.started
            session = self.sessions[path] = _Session(0 if new else stat.st_size)
            folder = self.initial_folder(path)
            if folder:
                session.label = Path(str(folder).rstrip('\\/')).name or None

    def poll(self):
        cfg = self.load_config()
        if self.started is None:
            self.started = time.time()
            self.scan(cfg)  # Snapshot every existing session's end.
            return
        if time.time() - self.last_scan >= RESCAN_SECONDS:
            self.scan(cfg)
        if not cfg.get('enabled'):
            self.pending.clear()
            for path, session in self.sessions.items():
                try:
                    session.offset = path.stat().st_size  # Unmuting skips the gap.
                except OSError:
                    pass
            return
        for path, session in list(self.sessions.items()):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == session.offset:
                continue
            if size < session.offset:
                session.offset = 0
            for record, session.offset in new_records(path, session.offset):
                self.handle(path, session, record, cfg)
        if self.pending:
            path, text = next(iter(self.pending.items()))
            if self.speak(text):
                self.pending.pop(path, None)

    def handle(self, path, session, record, cfg):
        try:
            parsed = self.parse(record, cfg)
        except AttributeError:
            return
        if parsed is None:
            return
        if parsed.prompt:
            session.last_prompt = parsed.prompt
        if parsed.folder:
            session.label = Path(str(parsed.folder).rstrip('\\/')).name or None
        fresh = record_age(record.get('timestamp')) <= STALE_SECONDS
        if parsed.prompt and fresh and self.prepare:
            self.prepare()
        if fresh and self.scenario:
            self._scenes(session, parsed)
        events = ([('prompt',)] if parsed.prompt else []) + parsed.events
        if parsed.narration:
            events.append(('narration',))
        if parsed.reply:
            events.append(('done',))
        for event in events:
            name = session.cues.feed(event, record.get('timestamp'))
            if name in ('succeeded', 'failed'):
                self._show(session, path, 'result', 'Tests passed' if name == 'succeeded' else 'Tests failed')
            if name == 'narrate':
                text = speech_text(parsed.narration, NARRATION_CHARS)
                if text and self.narrate:
                    self.narrate(text)
            elif name and self.cue:
                self.cue(name)
        if fresh:
            # Shown whether or not anything gets spoken: the line keeps up
            # with every step, speech only fills silences.
            if parsed.narration:
                self._show(session, path, 'note', speech_text(parsed.narration, 400))
            for step in parsed.steps:
                self._show(session, path, 'step', step)
            if any(e[0] == 'ask' for e in parsed.events):
                self._show(session, path, 'ask', 'Waiting for you')
            if parsed.reply:
                self._show(session, path, 'done', 'Reply ready')
                # The reply itself, for the subtitles (they show its opening).
                self._show(session, path, 'reply', speech_text(parsed.reply[1], 600))
        if parsed.reply and parsed.reply[0] not in self.seen:
            self.seen.add(parsed.reply[0])
            spoken = speech_text(parsed.reply[1], cfg.get('max_chars', 6000))
            if spoken:
                if path in self.pending:
                    log.info('%s voice relay skipping an older reply in %s', self.name, path.stem)
                # One reply per session waits; the voice drops the opener
                # when the reply already opens with a clip phrase.
                self.pending.pop(path, None)
                self.pending[path] = f'{opener(session.last_prompt)} {spoken}'

    def _scenes(self, session, parsed):
        scenes = session.scenes
        names = []
        if parsed.prompt:
            scenes.prompt()
            # The greeting after a long break, otherwise "target confirmed".
            names.append(greeting() or scenes.acknowledge())
        if parsed.narration:
            names.append(scenes.note(note_scenario(parsed.narration)))
        names += [scenes.tool(call_id, phase) for call_id, phase in parsed.phases]
        for event in parsed.events:
            if event[0] == 'result':
                names.append(scenes.result(event[1], event[2]))
            elif event[0] == 'ask':
                names.append(scenes.ask())
            elif event[0] == 'interrupted':
                names.append(scenes.interrupted())
        for name in filter(None, names):
            self.scenario(name, False)
        if parsed.reply:
            self.scenario(scenes.reply(parsed.reply[1]), True)

    def _show(self, session, path, kind, text):
        if self.activity and text:
            self.activity({'type': 'agent_activity', 'agent': self.name,
                           'session': session.label or path.parent.name,
                           'key': str(path), 'kind': kind, 'text': text})

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.poll()
            except Exception:
                log.exception('%s voice relay failed; will retry', self.name)
            self.stop_event.wait(0.75)


def toggle_main(config_path, defaults, label):
    """`python -m <relay module> --enable|--disable`."""
    import argparse
    parser = argparse.ArgumentParser()
    switch = parser.add_mutually_exclusive_group(required=True)
    switch.add_argument('--enable', action='store_true')
    switch.add_argument('--disable', action='store_true')
    args = parser.parse_args()
    cfg = load_json_config(config_path, defaults)
    cfg['enabled'] = args.enable
    temporary = Path(config_path).with_suffix('.tmp')
    temporary.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    temporary.replace(config_path)
    print(f'{label} voice enabled.' if args.enable else f'{label} voice muted.')
