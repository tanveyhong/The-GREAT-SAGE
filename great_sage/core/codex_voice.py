"""Read completed replies from one local Codex chat, without driving Codex."""
import json
import logging
from pathlib import Path
import re
import threading

from great_sage.voice.speakable import speakable, cap_for_speech

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parents[2] / 'codex_voice.json'


def completed_reply(record):
    payload = record.get('payload') or {}
    if record.get('type') != 'event_msg' or payload.get('type') != 'task_complete':
        return None
    text = payload.get('last_agent_message')
    if not isinstance(text, str) or not text.strip():
        return None
    return payload.get('turn_id') or record.get('timestamp'), text


def speech_text(text, limit=6000):
    # Coding stays visible in Codex; only prose is useful as spoken feedback.
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


class CodexVoiceRelay:
    def __init__(self, speak, config_path=CONFIG):
        self.speak = speak
        self.config_path = Path(config_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='codex-voice-relay', daemon=True)
        self.path = None
        self.offset = 0
        self.pending = []
        self.seen = set()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def clear(self):
        self.pending.clear()

    def poll(self):
        try:
            cfg = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            return
        path = Path(cfg.get('rollout_path', ''))
        if not cfg.get('enabled') or not path.is_file():
            self.pending.clear()
            self.path = None
            return
        if path != self.path:
            self.path = path
            self.offset = path.stat().st_size
            self.pending.clear()
            self.seen.clear()
            log.info('Codex voice relay attached to chat %s', cfg.get('thread_id', ''))
            return  # Start at the end; never replay old chat history.
        if path.stat().st_size < self.offset:
            self.offset = 0
        for record, self.offset in new_records(path, self.offset):
            try:
                reply = completed_reply(record)
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
                log.exception('Codex voice relay failed; will retry')
            self.stop_event.wait(0.75)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    switch = parser.add_mutually_exclusive_group(required=True)
    switch.add_argument('--enable', action='store_true')
    switch.add_argument('--disable', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    cfg['enabled'] = args.enable
    temporary = CONFIG.with_suffix('.tmp')
    temporary.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    temporary.replace(CONFIG)
    print('Codex voice enabled.' if args.enable else 'Codex voice muted.')
