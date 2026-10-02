"""Short recorded cues for what a relayed coding agent is doing.

The relays speak an agent's final reply. While it works - searching,
reading, running tests - narrating each step would always lag (every
spoken line costs seconds of translation and synthesis), so instead a
phase change plays one pre-recorded clip, instantly:

    begin      first tool call of a turn        "Beginning analysis."
    working    still at it, gaps doubling       "Analysing."
    succeeded  a test/build command passed      "Succeeded."
    failed     a test/build command failed      "Failed."
    needs_you  the agent asked Master something "Notice."

Relays turn their own log formats into the plain events CueTracker.feed
takes, so the rules live here once for Claude and Codex alike.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

CONFIG = Path(__file__).resolve().parents[2] / 'progress_cues.json'
DEFAULTS = {'enabled': True, 'interval_seconds': 45}

# A record older than this is history - typically a backlog read after a
# reply finished being spoken - and must not fire a cue now.
STALE_SECONDS = 15
# Never two cues closer than this; a burst of tool calls is one phase.
MIN_GAP_SECONDS = 4
# "Analysing." backs off from interval_seconds by doubling, to this cap.
MAX_WORKING_GAP = 300

# Commands whose pass/fail is worth hearing. Anything else failing (a grep
# that found nothing, a probe) is routine and stays silent.
TEST_COMMAND = re.compile(
    r'\b(pytest|unittest|tox|check_\w+\.py|build\.py|robustness_check\.py'
    r'|(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|build|lint)|cargo\s+(?:test|build)'
    r'|go\s+(?:test|build|vet)|dotnet\s+(?:test|build)|jest|vitest|tsc|mvn|gradle|make)\b',
    re.IGNORECASE)


def load_config(path=CONFIG):
    try:
        cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        cfg = {}
    return {**DEFAULTS, **cfg}


def record_age(timestamp, now=None):
    """Seconds since an ISO-8601 log timestamp; 0 if it cannot be read."""
    try:
        when = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
    except ValueError:
        return 0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now or time.time()) - when.timestamp()


class CueTracker:
    """Events in, cue names out. Events:

        ('prompt',)                 Master sent a new message
        ('tool', call_id, command)  a tool call; command is the shell text or None
        ('result', call_id, ok)     its result; ok is True, False or None (unknown)
        ('ask',)                    the agent is asking Master something
        ('done',)                   the turn's final reply arrived
    """

    def __init__(self, config_path=CONFIG, clock=time.time):
        self.config_path = config_path
        self.clock = clock
        self.started = False
        self.last_cue = 0.0
        self.commands = {}  # call_id -> command, for tests awaiting a result
        self.repeats = 0    # "Analysing."s so far this turn

    def feed(self, event, timestamp=None):
        cue = self._cue_for(event)
        if cue is None:
            return None
        cfg = load_config(self.config_path)
        now = self.clock()
        if not cfg.get('enabled') or (timestamp and record_age(timestamp, now) > STALE_SECONDS):
            return None
        urgent = cue in ('needs_you', 'failed', 'succeeded')
        if not urgent and now - self.last_cue < MIN_GAP_SECONDS:
            return None
        if cue == 'working':
            # Each repeat waits twice as long, up to MAX_WORKING_GAP: a
            # 30-minute turn at a flat 45s was dozens of "Analysing."s.
            gap = min(float(cfg.get('interval_seconds', 45)) * 2 ** self.repeats, MAX_WORKING_GAP)
            if now - self.last_cue < gap:
                return None
            self.repeats += 1
        self.last_cue = now
        return cue

    def _cue_for(self, event):
        kind = event[0]
        if kind == 'prompt':
            self.started = False
            self.repeats = 0
            self.commands.clear()
            return None
        if kind == 'done':
            self.started = False
            return None
        if kind == 'ask':
            return 'needs_you'
        if kind == 'tool':
            _, call_id, command = event
            if command and TEST_COMMAND.search(command):
                self.commands[call_id] = command
            if not self.started:
                self.started = True
                return 'begin'
            return 'working'
        if kind == 'result':
            _, call_id, ok = event
            if call_id in self.commands and ok is not None:
                del self.commands[call_id]
                return 'succeeded' if ok else 'failed'
        return None


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
    print('Progress cues enabled.' if args.enable else 'Progress cues muted.')
