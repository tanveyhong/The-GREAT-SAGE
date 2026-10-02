"""Voice every Codex session as it works, without driving Codex."""
import json
from pathlib import Path
import re

from great_sage.core.session_relay import (  # noqa: F401 - re-exported
    Parsed, SessionRelay, load_json_config, new_records, opener, speech_text, toggle_main)

CONFIG = Path(__file__).resolve().parents[2] / 'codex_voice.json'
DEFAULTS = {
    'enabled': True,
    'sessions_dir': str(Path.home() / '.codex' / 'sessions'),
    # "all" voices every Codex chat; "pinned" only rollout_path's.
    'follow': 'all',
    'max_chars': 6000,
}


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


def tool_steps(record):
    """'Running npm test'-style labels for the activity line."""
    payload = record.get('payload') or {}
    if record.get('type') != 'response_item' or payload.get('type') not in ('custom_tool_call', 'function_call'):
        return []
    name = payload.get('name', '')
    if name in ('sleep', 'wait', 'request_user_input_async'):
        return []
    if name == 'apply_patch':
        return ['Editing files']
    source = payload.get('input') or payload.get('arguments') or ''
    commands = _COMMAND.findall(source if isinstance(source, str) else '')
    if not commands:
        return ['Working']
    text = ' '.join(commands[0].replace('\\"', '"').split())
    return ['Running ' + (text if len(text) <= 56 else text[:55] + '…')]


def folder(record):
    """The chat's working folder, from its session and turn context records."""
    if record.get('type') in ('session_meta', 'turn_context'):
        return (record.get('payload') or {}).get('cwd')
    return None


def _message_text(payload):
    return ' '.join(block.get('text', '') for block in payload.get('content') or ()
                    if isinstance(block, dict)).strip()


def user_prompt(record):
    """The text Master typed into Codex, or None for injected context."""
    payload = record.get('payload') or {}
    if (record.get('type') != 'response_item' or payload.get('type') != 'message'
            or payload.get('role') != 'user'):
        return None
    text = _message_text(payload)
    # Environment context, app events and question replies arrive as tags.
    return text if text and not text.startswith('<') else None


def narration(record):
    """Codex's between-steps notes are assistant messages in the
    "commentary" phase; the reply itself is "final_answer"."""
    payload = record.get('payload') or {}
    if (record.get('type') != 'response_item' or payload.get('type') != 'message'
            or payload.get('role') != 'assistant' or payload.get('phase') != 'commentary'):
        return None
    return _message_text(payload) or None


def load_config(path=CONFIG):
    return load_json_config(path, DEFAULTS)


class CodexVoiceRelay(SessionRelay):
    name = 'Codex'

    def __init__(self, speak, config_path=CONFIG, cue=None, narrate=None, activity=None):
        super().__init__(speak, config_path, cue=cue, narrate=narrate, activity=activity)

    def load_config(self):
        return load_config(self.config_path)

    def session_files(self, cfg):
        if cfg.get('follow') == 'pinned':
            path = Path(cfg.get('rollout_path', ''))
            return [path] if path.is_file() else []
        return Path(cfg['sessions_dir']).rglob('rollout-*.jsonl')

    def initial_folder(self, path):
        try:
            with path.open('rb') as source:
                return folder(json.loads(source.readline()))
        except (OSError, ValueError, AttributeError):
            return None

    def parse(self, record, cfg):
        return Parsed(prompt=user_prompt(record),
                      reply=completed_reply(record),
                      narration=narration(record),
                      events=tool_events(record),
                      steps=tool_steps(record),
                      folder=folder(record))


if __name__ == '__main__':
    toggle_main(CONFIG, DEFAULTS, 'Codex')
