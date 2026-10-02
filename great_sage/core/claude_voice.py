"""Read completed replies from the newest Claude desktop session, without driving Claude."""
import json
import logging
from pathlib import Path
import threading
import time

from great_sage.core.codex_voice import new_records, opener, speech_text
from great_sage.core.progress_cues import CueTracker

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parents[2] / 'claude_voice.json'
DEFAULTS = {
    'enabled': True,
    'projects_dir': str(Path.home() / '.claude' / 'projects'),
    'entrypoint': 'claude-desktop',  # '' speaks CLI sessions too.
    'max_chars': 6000,
}


def user_prompt(record):
    """The text Master typed, or None for tool results and system notices."""
    if record.get('type') != 'user' or record.get('isSidechain') or record.get('isMeta'):
        return None
    content = (record.get('message') or {}).get('content')
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content):
            return None
        content = ' '.join(b.get('text', '') for b in content
                           if isinstance(b, dict) and b.get('type') == 'text')
    if not isinstance(content, str) or not content.strip() or content.lstrip().startswith('<'):
        return None
    return content.strip()


# Tools that stop and wait for Master's answer.
_ASKING_TOOLS = {'AskUserQuestion', 'ExitPlanMode'}
_SHELL_TOOLS = {'Bash', 'PowerShell'}


def tool_events(record):
    """Progress events (see progress_cues.CueTracker) in one log record."""
    if record.get('isSidechain'):
        return []
    content = (record.get('message') or {}).get('content')
    if not isinstance(content, list):
        return []
    events = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'tool_use':
            name = block.get('name', '')
            if name in _ASKING_TOOLS:
                events.append(('ask',))
            else:
                command = (block.get('input') or {}).get('command') if name in _SHELL_TOOLS else None
                events.append(('tool', block.get('id'), command if isinstance(command, str) else None))
        elif block.get('type') == 'tool_result':
            events.append(('result', block.get('tool_use_id'), not block.get('is_error')))
    return events


def completed_reply(record, entrypoint='claude-desktop'):
    message = record.get('message') or {}
    if (record.get('type') != 'assistant' or record.get('isSidechain')
            or message.get('stop_reason') != 'end_turn'
            or message.get('model') == '<synthetic>'):
        return None  # Tool-call narration, sub-agents, and local error stubs.
    if entrypoint and record.get('entrypoint') != entrypoint:
        return None
    content = message.get('content')
    if isinstance(content, str):
        text = content
    else:
        text = '\n\n'.join(block.get('text', '') for block in content or ()
                           if isinstance(block, dict) and block.get('type') == 'text')
    if not text.strip():
        return None
    return record.get('uuid') or message.get('id'), text


def load_config(path=CONFIG):
    try:
        cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        cfg = {}
    return {**DEFAULTS, **cfg}


def _created(stat):
    return getattr(stat, 'st_birthtime', stat.st_ctime)


class ClaudeVoiceRelay:
    def __init__(self, speak, config_path=CONFIG, cue=None):
        self.speak = speak
        self.cue = cue  # Plays a progress cue by name; see progress_cues.
        self.cues = CueTracker()
        self.config_path = Path(config_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='claude-voice-relay', daemon=True)
        self.started = None
        self.offsets = {}  # Per session file, so switching back never replays.
        self.path = None
        self.pending = []
        self.seen = set()
        self.last_prompt = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def clear(self):
        self.pending.clear()

    def sessions(self, cfg):
        # Top-level files only: sub-agent logs live in per-session folders.
        return Path(cfg['projects_dir']).glob('*/*.jsonl')

    def newest(self, cfg):
        best, best_mtime = None, None
        for path in self.sessions(cfg):
            try:
                stat = path.stat()
            except OSError:
                continue
            if path not in self.offsets:
                # Sessions begun after startup are read from the top;
                # older ones from their current end, never their history.
                self.offsets[path] = 0 if _created(stat) >= self.started else stat.st_size
            if best_mtime is None or stat.st_mtime > best_mtime:
                best, best_mtime = path, stat.st_mtime
        return best

    def poll(self):
        cfg = load_config(self.config_path)
        if self.started is None:
            self.started = time.time()
            self.newest(cfg)  # Snapshot every existing session's end.
            return
        if not cfg.get('enabled'):
            self.pending.clear()
            self.newest(cfg)  # Keep offsets current so unmuting skips the gap.
            for path in self.offsets:
                try:
                    self.offsets[path] = path.stat().st_size
                except OSError:
                    pass
            return
        path = self.newest(cfg)
        if path is None:
            return
        if path != self.path:
            self.path = path
            log.info('Claude voice relay following session %s', path.stem)
        if path.stat().st_size < self.offsets[path]:
            self.offsets[path] = 0
        for record, self.offsets[path] in new_records(path, self.offsets[path]):
            try:
                prompt = user_prompt(record)
                if prompt:
                    self.last_prompt = prompt
                reply = completed_reply(record, cfg.get('entrypoint', ''))
                entry = cfg.get('entrypoint', '')
                if not entry or record.get('entrypoint') == entry:
                    events = ([('prompt',)] if prompt else []) + tool_events(record)
                    self._feed_cues(events + ([('done',)] if reply else []), record.get('timestamp'))
            except AttributeError:
                continue
            if reply and reply[0] not in self.seen:
                self.seen.add(reply[0])
                spoken = speech_text(reply[1], cfg.get('max_chars', 6000))
                if spoken:
                    # The voice drops this when the reply already opens
                    # with a clip phrase of its own.
                    self.pending.append(f'{opener(self.last_prompt)} {spoken}')
        if len(self.pending) > 1:
            # Replies that piled up while one was spoken are stale by now.
            log.info('Voice relay skipping %d older replies', len(self.pending) - 1)
            del self.pending[:-1]
        if self.pending and self.speak(self.pending[0]):
            self.pending.pop(0)

    def _feed_cues(self, events, timestamp):
        for event in events:
            name = self.cues.feed(event, timestamp)
            if name and self.cue:
                self.cue(name)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.poll()
            except Exception:
                log.exception('Claude voice relay failed; will retry')
            self.stop_event.wait(0.75)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    switch = parser.add_mutually_exclusive_group(required=True)
    switch.add_argument('--enable', action='store_true')
    switch.add_argument('--disable', action='store_true')
    args = parser.parse_args()
    cfg = load_config()
    cfg['enabled'] = args.enable
    temporary = CONFIG.with_suffix('.tmp')
    temporary.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    temporary.replace(CONFIG)
    print('Claude voice enabled.' if args.enable else 'Claude voice muted.')
