"""Voice every Claude desktop session as it works, without driving Claude."""
from pathlib import Path

from great_sage.core.agent_scenarios import phase_of
from great_sage.core.session_relay import (
    Parsed, SessionRelay, load_json_config, toggle_main)

CONFIG = Path(__file__).resolve().parents[2] / 'claude_voice.json'
DEFAULTS = {
    'enabled': True,
    'projects_dir': str(Path.home() / '.claude' / 'projects'),
    'entrypoint': 'claude-desktop',  # '' speaks CLI sessions too.
    'max_chars': 6000,
}

# Tools that stop and wait for Master's answer.
_ASKING_TOOLS = {'AskUserQuestion', 'ExitPlanMode'}
_SHELL_TOOLS = {'Bash', 'PowerShell'}


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


def _short(text, limit=60):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def _base(path):
    return Path(str(path)).name if path else ''


def tool_steps(record):
    """'Editing WorkflowService.php'-style labels for the activity line."""
    if record.get('isSidechain') or record.get('type') != 'assistant':
        return []
    content = (record.get('message') or {}).get('content')
    steps = []
    for block in content if isinstance(content, list) else ():
        if not isinstance(block, dict) or block.get('type') != 'tool_use':
            continue
        name, args = block.get('name', ''), block.get('input') or {}
        if name in _SHELL_TOOLS:
            steps.append(_short(args.get('description') or 'Running ' + str(args.get('command', ''))))
        elif name == 'Read':
            steps.append('Reading ' + _base(args.get('file_path')))
        elif name in ('Edit', 'MultiEdit', 'NotebookEdit'):
            steps.append('Editing ' + _base(args.get('file_path') or args.get('notebook_path')))
        elif name == 'Write':
            steps.append('Writing ' + _base(args.get('file_path')))
        elif name == 'Grep':
            steps.append(_short(f"Searching '{args.get('pattern', '')}'"))
        elif name == 'Glob':
            steps.append(_short('Finding ' + str(args.get('pattern', ''))))
        elif name in ('WebSearch', 'WebFetch'):
            steps.append('Searching the web')
        elif name in ('Agent', 'Task'):
            steps.append(_short('Sub-agent: ' + str(args.get('description', ''))))
        elif name not in _ASKING_TOOLS:
            steps.append(_short(name.replace('mcp__', '').replace('__', ' ')))
    return steps


def tool_phases(record):
    """(call_id, scenario phase) for each tool call in the record."""
    if record.get('isSidechain') or record.get('type') != 'assistant':
        return []
    content = (record.get('message') or {}).get('content')
    phases = []
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and block.get('type') == 'tool_use':
            args = block.get('input') or {}
            command = args.get('command') if block.get('name') in _SHELL_TOOLS else None
            phases.append((block.get('id'), phase_of(block.get('name'), command)))
    return phases


def _assistant_text(record, stop_reason):
    message = record.get('message') or {}
    if (record.get('type') != 'assistant' or record.get('isSidechain')
            or message.get('stop_reason') != stop_reason
            or message.get('model') == '<synthetic>'):
        return None  # Sub-agents and local error stubs never speak.
    content = message.get('content')
    if isinstance(content, str):
        text = content
    else:
        text = '\n\n'.join(block.get('text', '') for block in content or ()
                           if isinstance(block, dict) and block.get('type') == 'text')
    return text if text.strip() else None


def completed_reply(record, entrypoint='claude-desktop'):
    """(id, text) for a turn's final reply."""
    if entrypoint and record.get('entrypoint') != entrypoint:
        return None
    text = _assistant_text(record, 'end_turn')
    if text is None:
        return None
    return record.get('uuid') or (record.get('message') or {}).get('id'), text


def narration(record):
    """The note Claude writes just before a tool call ("Now running the tests.")."""
    return _assistant_text(record, 'tool_use')


def load_config(path=CONFIG):
    return load_json_config(path, DEFAULTS)


class ClaudeVoiceRelay(SessionRelay):
    name = 'Claude'

    def __init__(self, speak, config_path=CONFIG, **hooks):
        super().__init__(speak, config_path, **hooks)

    def load_config(self):
        return load_config(self.config_path)

    def session_files(self, cfg):
        # Top-level files only: sub-agent logs live in per-session folders.
        return Path(cfg['projects_dir']).glob('*/*.jsonl')

    def parse(self, record, cfg):
        entry = cfg.get('entrypoint', '')
        # Messages carry where the session runs; other record kinds do not.
        if entry and record.get('type') in ('user', 'assistant') and record.get('entrypoint') != entry:
            return None
        return Parsed(prompt=user_prompt(record),
                      reply=completed_reply(record, ''),
                      narration=narration(record),
                      events=tool_events(record),
                      steps=tool_steps(record),
                      folder=record.get('cwd'),
                      phases=tool_phases(record))


if __name__ == '__main__':
    toggle_main(CONFIG, DEFAULTS, 'Claude')
