"""Replay a past Claude or Codex session into the running Companion, faster.

    py -m great_sage.core.replay                      # newest Claude session
    py -m great_sage.core.replay --codex              # newest Codex chat
    py -m great_sage.core.replay path\\to\\log.jsonl --speed 20 --minutes 5

Nothing in the app changes: the relays already watch the session folders.
This copies a session, record by record, into a NEW log in a folder they
watch, with each record's timestamp moved to now and the gaps between
records shortened by --speed (and capped). To the Companion it is a live
session - scenario clips, subtitles and the activity line all react as
they would to real work - so changes to any of them can be seen and heard
without waiting for an agent to do something. The copy is deleted after.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

CLAUDE_DIR = Path.home() / '.claude' / 'projects'
CODEX_DIR = Path.home() / '.codex' / 'sessions'
REPLAY_FOLDER = 'great-sage-replay'   # never a real project's name
MAX_GAP = 4.0                         # seconds: no long silences in a replay


def _newest(paths, skip=None):
    paths = [p for p in paths if skip is None or skip not in p.parts]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _when(record):
    stamp = record.get('timestamp')
    try:
        return datetime.fromisoformat(str(stamp).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def replay(source, codex=False, speed=10.0, minutes=None):
    records = []
    for line in source.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    if minutes:
        # Only the last N minutes of the session: the recent turns.
        times = [t for t in map(_when, records) if t]
        if times:
            cutoff = max(times) - minutes * 60
            first_kept = next((i for i, r in enumerate(records) if (_when(r) or 0) >= cutoff), 0)
            # Keep a Codex chat's session record so its folder name shows.
            head = [records[0]] if codex and first_kept else []
            records = head + records[first_kept:]
    if codex:
        target_dir = CODEX_DIR / REPLAY_FOLDER
        target = target_dir / f'rollout-replay-{int(time.time())}.jsonl'
    else:
        target_dir = CLAUDE_DIR / REPLAY_FOLDER
        target = target_dir / f'replay-{int(time.time())}.jsonl'
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f'Replaying {len(records)} records from {source.name} at {speed:g}x into {target}')
    previous = None
    try:
        with target.open('w', encoding='utf-8') as out:
            for record in records:
                then = _when(record)
                if previous is not None and then is not None:
                    time.sleep(min(max(0.0, then - previous) / speed, MAX_GAP))
                if then is not None:
                    previous = then
                    record['timestamp'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
        time.sleep(8)   # let the last clip and subtitles finish
    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        try:
            target.unlink()
        except OSError:
            pass
    print('Replay finished.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('log', nargs='?', help='a session .jsonl (default: the newest)')
    parser.add_argument('--codex', action='store_true', help='replay a Codex chat')
    parser.add_argument('--speed', type=float, default=10.0, help='how many times faster (default 10)')
    parser.add_argument('--minutes', type=float, default=10.0,
                        help='replay only the last N minutes of the session (default 10; 0 = all)')
    args = parser.parse_args()
    if args.log:
        source = Path(args.log)
        codex = args.codex or source.name.startswith('rollout-')
    elif args.codex:
        source, codex = _newest(list(CODEX_DIR.rglob('rollout-*.jsonl')), REPLAY_FOLDER), True
    else:
        source, codex = _newest(list(CLAUDE_DIR.glob('*/*.jsonl')), REPLAY_FOLDER), False
    if source is None or not source.is_file():
        raise SystemExit('No session log found to replay.')
    replay(source, codex=codex, speed=max(1.0, args.speed), minutes=args.minutes or None)


if __name__ == '__main__':
    main()
