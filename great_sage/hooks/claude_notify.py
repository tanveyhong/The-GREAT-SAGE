"""Claude Code Notification hook -> Great Sage.

Claude Code runs this (see ~/.claude/settings.json, "Notification") when it
needs Master: a permission prompt ("Claude needs your permission to use
Bash"), or waiting idle for input. Those never appear in the session log,
so the Companion could not hear them; this hands them over the local
websocket the HUD already uses.

It must never slow Claude down or show an error: one message, a 2-second
limit, and it always exits 0 - Great Sage simply not running is normal.
Run by full path (hooks start in the session's folder, not this project's).
"""
import json
import os
import sys

URL = os.environ.get('GREAT_SAGE_WS', 'ws://127.0.0.1:8765')


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    message = {
        'type': 'agent_notification',
        'agent': 'Claude',
        'message': str(payload.get('message') or ''),
        'notification_type': str(payload.get('notification_type') or ''),
        'cwd': str(payload.get('cwd') or ''),
        'transcript_path': str(payload.get('transcript_path') or ''),
    }
    try:
        from websockets.sync.client import connect
        with connect(URL, open_timeout=2, close_timeout=1) as ws:
            ws.send(json.dumps(message))
    except Exception:
        pass   # Great Sage is not running, or busy: nothing to do.


if __name__ == '__main__':
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
