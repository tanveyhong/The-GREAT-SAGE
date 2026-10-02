"""Pre-made Great Sage lines for what a coding agent is doing.

With translation off, the relays do not synthesize anything: each change
of scenario plays a clip made once, ahead of time, by
`py -m great_sage.voice.make_agent_clips`. No translation model and no
XTTS run while Master codes - the GPU and RAM cost of live speech is gone.

The scenarios come from analysing 476 real Claude and Codex sessions
(2026-10-02): reading ~14.9k calls, searching ~12.4k, editing ~10.1k,
browser 2.1k, git 1.9k, database 1.2k, tests/builds 768, servers 280,
installs 180; of 6035 final replies 35% report passing tests, 29% a fix,
29% something built, 19% mention a failure, 6% something impossible,
4% a restart, 2% end on a question.

Each line opens with a real recorded clip (koku 告 / kai 解), then XTTS
speaks the rest; the generator joins them into one file.
"""
import random
import re
import time
from pathlib import Path

from great_sage.core.progress_cues import TEST_COMMAND

CLIP_DIR = Path(__file__).resolve().parents[2] / 'voice_lines' / 'agent'

# name -> (recorded opener or None, [(Japanese, English caption), ...])
CATALOGUE = {
    # --- while working: one clip per change of phase ---
    'investigate': ('koku', [('対象の解析を開始します。', 'Beginning analysis of the target.'),
                             ('コードを調査しています。', 'Examining the code.')]),
    'search': ('koku', [('該当箇所を検索しています。', 'Searching for the relevant code.')]),
    'edit': ('koku', [('修正を適用しています。', 'Applying the changes.'),
                      ('コードを書き換えています。', 'Rewriting the code.')]),
    'test_run': ('koku', [('検証を実行します。', 'Running verification.')]),
    'browser': ('koku', [('画面の状態を確認しています。', 'Checking the screen.')]),
    'database': ('koku', [('データベースを照会しています。', 'Querying the database.')]),
    'git': ('koku', [('変更を記録しています。', 'Recording the changes.')]),
    'install': ('koku', [('必要な構成要素を導入しています。', 'Installing what is needed.')]),
    'server': ('koku', [('サーバーを起動しています。', 'Starting the server.')]),
    'subagent': ('koku', [('並列解析を開始します。', 'Starting parallel analysis.')]),
    'working': (None, [('解析を継続しています。', 'Analysis continues.')]),
    # --- results ---
    'test_pass': ('kai', [('検証に成功しました。', 'Verification succeeded.'),
                          ('すべての確認を通過しました。', 'All checks passed.')]),
    'test_fail': ('koku', [('検証に失敗しました。原因を解析します。',
                            'Verification failed. Analysing the cause.')]),
    'push': ('koku', [('変更を送信しました。', 'Changes sent.')]),
    'ask': ('koku', [('マスター、判断を求めます。', 'Master, your decision is needed.')]),
    # --- the final reply, by what it says ---
    'done': ('kai', [('作業が完了しました。', 'Work complete.'),
                     ('処理を完了しました。', 'Task complete.')]),
    'done_verified': ('kai', [('作業完了。検証にも成功しました。', 'Work complete, and verified.')]),
    'fixed': ('kai', [('修正が完了しました。', 'The fix is complete.')]),
    'problem': ('koku', [('問題が発生しました。報告を確認してください。',
                          'A problem occurred. Please check the report.')]),
    'question': ('koku', [('マスター、回答をお待ちしています。', 'Master, awaiting your answer.')]),
    'restart': ('koku', [('再起動が必要です。', 'A restart is required.')]),
    'reply': ('koku', [('報告があります。', 'I have a report.')]),
}

PHASES = {'investigate', 'search', 'edit', 'test_run', 'browser', 'database', 'git',
          'install', 'server', 'subagent'}

_GIT_PUSH = re.compile(r'\bgit\s+push\b')
_GIT = re.compile(r'\bgit\s+(commit|add|status|diff|log|fetch|pull|checkout|switch|branch|merge|rebase|stash)\b')
_INSTALL = re.compile(r'\b(pip|npm|pnpm|yarn|winget|choco|composer)\s+(install|add|i)\b', re.I)
_SERVER = re.compile(r'\b(artisan\s+serve|npm\s+run\s+dev|vite\b|flask\s+run|uvicorn|preview_start)', re.I)
_DATABASE = re.compile(r'\b(migrate|mysql|psql|sqlite3?|tinker|db:\w+)\b', re.I)
_SEARCH = re.compile(r'\b(grep|rg|findstr|Select-String|Get-ChildItem|find)\b', re.I)
_READ = re.compile(r'\b(cat|head|tail|sed\s+-n|Get-Content|type)\b', re.I)


def phase_of(tool, command=None):
    """Scenario for one tool call: its tool name, and shell text if any."""
    tool = tool or ''
    if tool in ('Grep', 'Glob'):
        return 'search'
    if tool == 'Read':
        return 'investigate'
    if tool in ('Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'apply_patch'):
        return 'edit'
    if tool in ('Agent', 'Task'):
        return 'subagent'
    if 'Browser' in tool or 'chrome' in tool:
        return 'browser'
    if command:
        if _GIT_PUSH.search(command):
            return 'push'
        if TEST_COMMAND.search(command):
            return 'test_run'
        for phase, pattern in (('git', _GIT), ('install', _INSTALL), ('server', _SERVER),
                               ('database', _DATABASE), ('search', _SEARCH), ('investigate', _READ)):
            if pattern.search(command):
                return phase
    return None


def reply_scenario(text):
    """The final reply's scenario, from what it reports."""
    t = (text or '').strip().lower()
    if t.endswith('?'):
        return 'question'
    if re.search(r'\brestart', t) and re.search(r'\b(need|requires?|must)\b', t):
        return 'restart'
    passed = re.search(r'\b(tests? pass|all (checks|tests) pass|passes|passed|all green)\b', t)
    fixed = re.search(r'\bfix(ed|es)?\b', t)
    built = re.search(r'\b(added|built|created|implemented|done)\b', t)
    if re.search(r"\b(can't|cannot|couldn't|unable to|blocked)\b", t) and not (fixed or passed):
        return 'problem'
    if re.search(r'\b(fail(ed|s|ing)?|error|broke|crash)', t) and not (fixed or passed):
        return 'problem'
    if passed and (fixed or built):
        return 'done_verified'
    if fixed:
        return 'fixed'
    if built or passed:
        return 'done'
    return 'reply'


def clip_files(name):
    """The generated clips for a scenario, in variant order."""
    # Exactly <name>_<n>.wav: a bare glob would let "done" take done_verified_0.
    variant = re.compile(re.escape(name) + r'_\d+\.wav$')
    return sorted(p for p in CLIP_DIR.glob(f'{name}_*.wav') if variant.match(p.name))


class ScenarioTracker:
    """Per-session: tool phases, results and replies in, scenario names out.

    A phase plays only when it changes, never closer than PHASE_GAP after
    the last clip, and a repeat of the same phase waits REPEAT_GAP - twenty
    searches in a row are one "searching". Results, questions and replies
    always play."""
    PHASE_GAP = 10
    REPEAT_GAP = 90
    WORKING_GAP = 120

    def __init__(self, clock=time.time):
        self.clock = clock
        self.last_phase = None
        self.last_at = 0.0
        self.phase_at = {}
        self.pending = {}  # call_id -> phase, for test runs and pushes

    def prompt(self):
        self.last_phase = None
        self.phase_at.clear()
        self.pending.clear()

    def tool(self, call_id, phase):
        if phase in ('test_run', 'push'):
            self.pending[call_id] = phase
        if phase == 'push':
            return None  # Announced when it succeeds.
        now = self.clock()
        if phase is None or now - self.last_at < self.PHASE_GAP:
            return None
        if phase == self.last_phase or now - self.phase_at.get(phase, 0) < self.REPEAT_GAP:
            if now - self.last_at >= self.WORKING_GAP:
                return self._emit('working', now)
            return None
        self.last_phase = phase
        self.phase_at[phase] = now
        return self._emit(phase, now)

    def result(self, call_id, ok):
        phase = self.pending.pop(call_id, None)
        if phase is None or ok is None:
            return None
        if phase == 'push':
            return self._emit('push', self.clock()) if ok else None
        return self._emit('test_pass' if ok else 'test_fail', self.clock())

    def ask(self):
        return self._emit('ask', self.clock())

    def reply(self, text):
        self.last_phase = None
        return self._emit(reply_scenario(text), self.clock())

    def _emit(self, name, now):
        self.last_at = now
        return name


_last_pick = {}


def pick_clip(name):
    """A variant of the scenario's clip, never the same one twice running."""
    files = clip_files(name)
    if not files:
        return None
    choices = [f for f in files if f != _last_pick.get(name)] or files
    choice = random.choice(choices)
    _last_pick[name] = choice
    return choice


def caption(name, path):
    """The English caption for a generated clip."""
    try:
        index = int(Path(path).stem.rsplit('_', 1)[1])
        return CATALOGUE[name][1][index][1]
    except (KeyError, IndexError, ValueError):
        return None
