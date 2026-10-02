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


_EXIT_CODE = re.compile(r'"exit_code"\s*:\s*(-?\d+)')
_COMMAND = re.compile(r'\bcmd\s*:\s*"((?:[^"\\]|\\.)*)"')


def _output_text(output):
    if isinstance(output, list):
        return ' '.join(b.get('text', '') for b in output if isinstance(b, dict))
    return output if isinstance(output, str) else ''


def tool_events(record):
    """Progress events (see progress_cues.CueTracker) in one log record.

    Codex runs shell commands inside an `exec` script as
    tools.exec_command({cmd: "..."}), and reports "exit_code": N in the
    output, so both are read out of the text."""
    payload = record.get('payload') or {}
    kind = payload.get('type')
    if record.get('type') != 'response_item':
        return []
    if kind in ('custom_tool_call', 'function_call'):
        if payload.get('name') == 'request_user_input_async':
            return [('ask',)]
        if payload.get('name') in ('sleep', 'wait'):
            return []  # Codex idling for a command, not a new step.
        source = payload.get('input') or payload.get('arguments') or ''
        commands = _COMMAND.findall(source if isinstance(source, str) else '')
        return [('tool', payload.get('call_id'), ' ; '.join(commands) or None)]
    if kind in ('custom_tool_call_output', 'function_call_output'):
        codes = [int(c) for c in _EXIT_CODE.findall(_output_text(payload.get('output')))]
        return [('result', payload.get('call_id'), all(c == 0 for c in codes) if codes else None)]
    return []


def user_prompt(record):
    """The text Master typed into Codex, or None for injected context."""
    payload = record.get('payload') or {}
    if (record.get('type') != 'response_item' or payload.get('type') != 'message'
            or payload.get('role') != 'user'):
        return None
    text = ' '.join(block.get('text', '') for block in payload.get('content') or ()
                    if isinstance(block, dict) and block.get('type') == 'input_text').strip()
    # Environment context, app events and question replies arrive as tags.
    return text if text and not text.startswith('<') else None


_QUESTION_START = re.compile(
    r'^(what|why|how|is|are|was|were|can|could|does|do|did|should|which|who|'
    r'where|when|will|would|shall|has|have)\b', re.IGNORECASE)


def opener(prompt):
    """Great Sage's two openers: 解 "Answer." for a question, else 告 "Notice."."""
    if prompt and ('?' in prompt or _QUESTION_START.match(prompt)):
        return 'Answer.'
    return 'Notice.'


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
    def __init__(self, speak, config_path=CONFIG, cue=None):
        from great_sage.core.progress_cues import CueTracker
        self.speak = speak
        self.cue = cue  # Plays a progress cue by name; see progress_cues.
        self.cues = CueTracker()
        self.config_path = Path(config_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='codex-voice-relay', daemon=True)
        self.path = None
        self.offset = 0
        self.pending = []
        self.seen = set()
        self.last_prompt = None

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
                prompt = user_prompt(record)
                reply = completed_reply(record)
            except AttributeError:
                continue
            if prompt:
                self.last_prompt = prompt
            events = ([('prompt',)] if prompt else []) + tool_events(record)
            for event in events + ([('done',)] if reply else []):
                name = self.cues.feed(event, record.get('timestamp'))
                if name and self.cue:
                    self.cue(name)
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
