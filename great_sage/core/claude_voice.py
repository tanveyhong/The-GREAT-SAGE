"""Read completed replies from the newest Claude desktop session, without driving Claude."""
import json
import logging
from pathlib import Path
import threading
import time

from great_sage.core.codex_voice import new_records, speech_text

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parents[2] / 'claude_voice.json'
DEFAULTS = {
    'enabled': True,
    'projects_dir': str(Path.home() / '.claude' / 'projects'),
    'entrypoint': 'claude-desktop',  # '' speaks CLI sessions too.
    'max_chars': 6000,
}


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
    def __init__(self, speak, config_path=CONFIG):
        self.speak = speak
        self.config_path = Path(config_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='claude-voice-relay', daemon=True)
        self.started = None
        self.offsets = {}  # Per session file, so switching back never replays.
        self.path = None
        self.pending = []
        self.seen = set()

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
                reply = completed_reply(record, cfg.get('entrypoint', ''))
            except AttributeError:
                continue
            if reply and reply[0] not in self.seen:
                self.seen.add(reply[0])
                spoken = speech_text(reply[1], cfg.get('max_chars', 6000))
                if spoken:
                    self.pending.append(spoken)
        if len(self.pending) > 1:
            # Replies that piled up while one was spoken are stale by now.
            log.info('Voice relay skipping %d older replies', len(self.pending) - 1)
            del self.pending[:-1]
        if self.pending and self.speak(self.pending[0]):
            self.pending.pop(0)

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
