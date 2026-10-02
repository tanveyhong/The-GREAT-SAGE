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

# name -> (recorded opener or None, [(Japanese, English caption), ...]).
# A line '@<file>' is a real recording from voice_lines/ used as-is - those
# beat generated speech wherever one fits, so they lead the variants.
CATALOGUE = {
    # --- while working: one clip per change of phase ---
    'acknowledge': (None, [('@taishou_kakunin', 'Target confirmed.'), ('@ryo', 'Understood.')]),
    'investigate': ('koku', [('@kaiseki_kaishi', 'Understood. Beginning analysis.'),
                             ('@kaiseki_kantei', 'Running appraisal.'),
                             ('対象の解析を開始します。', 'Beginning analysis of the target.'),
                             ('コードを調査しています。', 'Examining the code.'),
                             ('関連する処理を追跡しています。', 'Tracing the related code.')]),
    'search': ('koku', [('該当箇所を検索しています。', 'Searching for the relevant code.'),
                        ('参照箇所を洗い出しています。', 'Finding every reference.')]),
    'edit': ('koku', [('修正を適用しています。', 'Applying the changes.'),
                      ('コードを書き換えています。', 'Rewriting the code.'),
                      ('変更を反映しています。', 'Putting the change in place.')]),
    'test_run': ('koku', [('検証を実行します。', 'Running verification.')]),
    # From the agents' own notes (15,589 notes, 2026-10-02): UI work 37%,
    # backend 18%, builds 10%, lint/type checks 4%, planning 4%, refactors
    # 3%, security 2%; turning points: found the cause 7%, an error 4%,
    # approval 3%, retrying 2%, reverting 2%.
    'ui': ('koku', [('画面の構成を調整しています。', 'Adjusting the screen layout.'),
                    ('表示の細部を整えています。', 'Refining the display.')]),
    'backend': ('koku', [('サーバー側の処理を解析しています。', 'Analysing the server-side logic.')]),
    'lint': ('koku', [('静的解析を実行します。', 'Running static analysis.')]),
    'plan': ('koku', [('作業手順を立案しています。', 'Planning the steps.')]),
    'refactor': ('koku', [('構造を整理しています。', 'Tidying the structure.')]),
    'security': ('koku', [('安全性を確認しています。', 'Checking the security.')]),
    'restarting': ('koku', [('再起動を実行します。', 'Restarting.')]),
    'found_cause': ('kai', [('原因を特定しました。', 'The cause is identified.'),
                            ('問題の発生源を突き止めました。', 'Found where the problem comes from.')]),
    'error_seen': ('koku', [('エラーを検知しました。', 'An error was detected.')]),
    'retry': ('koku', [('再試行します。', 'Trying again.')]),
    'revert': ('koku', [('変更を元に戻します。', 'Reverting the change.')]),
    'approval': ('koku', [('マスター、承認が必要です。', 'Master, your approval is needed.')]),
    # Turns run long: median 4 min, but 27% pass 10 min and 6% pass 30.
    'long_work': ('koku', [('解析が長引いています。もう少々お待ちください。',
                            'This is taking a while. Please wait a little longer.')]),
    'very_long': ('koku', [('大規模な作業です。完了まで継続します。',
                            'A large task. Continuing until it is done.')]),
    # Session: Master's first prompt after a long break, and late nights.
    'greeting_morning': ('kidou', [('', 'Good morning, Master.')]),
    'greeting_return': (None, [('お帰りなさいませ、マスター。', 'Welcome back, Master.')]),
    'late_night': ('koku', [('夜も更けています。ご無理はなさらぬよう。',
                             'It is late. Please do not overwork yourself.')]),
    'browser': ('koku', [('画面の状態を確認しています。', 'Checking the screen.')]),
    'database': ('koku', [('データベースを照会しています。', 'Querying the database.')]),
    'git': ('koku', [('変更を記録しています。', 'Recording the changes.')]),
    'install': ('koku', [('必要な構成要素を導入しています。', 'Installing what is needed.')]),
    'server': ('koku', [('サーバーを起動しています。', 'Starting the server.')]),
    'subagent': ('koku', [('並列解析を開始します。', 'Starting parallel analysis.')]),
    'working': (None, [('@kaiseki_chuu', 'Analysing.'), ('解析を継続しています。', 'Analysis continues.')]),
    'danger': (None, [('@seizon_kakuritsu', 'Survival odds drop sharply in a fight. Evasion recommended.')]),
    'missing': (None, [('@mishutoku', 'That skill is not yet acquired.')]),
    'interrupted': (None, [('@kaiseki_chuudan', 'Understood. Analysis interrupted.')]),
    # --- results ---
    'test_pass': ('kai', [('@seiko_shimashita', 'Success.'),
                          ('検証に成功しました。', 'Verification succeeded.'),
                          ('すべての確認を通過しました。', 'All checks passed.'),
                          ('問題は検出されませんでした。', 'No problems found.')]),
    'test_fail': ('koku', [('@shippai_shimashita', 'Failed.'),
                           ('検証に失敗しました。原因を解析します。',
                            'Verification failed. Analysing the cause.')]),
    'push': ('koku', [('変更を送信しました。', 'Changes sent.'), ('@seiko_shimashita', 'Success.')]),
    'ask': ('koku', [('マスター、判断を求めます。', 'Master, your decision is needed.')]),
    # --- the final reply, by what it says ---
    'done': ('kai', [('@taishou_kanryou', 'Analysis complete. Target information recorded.'),
                     ('@kaiseki_shuuryou', 'Ending appraisal.'),
                     ('作業が完了しました。', 'Work complete.'),
                     ('処理を完了しました。', 'Task complete.'),
                     ('ご依頼の作業を終えました。', 'Your request is done.')]),
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


# Commands that destroy work or data: Great Sage's "evasion recommended".
_DANGER = re.compile(r'\brm\s+-\w*r\w*f|\bgit\s+(reset\s+--hard|clean\s+-\w*f|push\s+(-f\b|--force))'
                     r'|\bdrop\s+(table|database)\b|\btruncate\s+table\b|\bRemove-Item\b.*-Recurse.*-Force'
                     r'|\bdel\s+/s\b|^\s*format\s+[a-z]:', re.I)
_MISSING = re.compile(r"\b(not installed|isn't installed|is not available|isn't available|no access to|"
                      r"command not found|not recognized as)\b", re.I)


def phase_of(tool, command=None):
    """Scenario for one tool call: its tool name, and shell text if any."""
    tool = tool or ''
    if command and _DANGER.search(command):
        return 'danger'
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
    if _MISSING.search(t):
        return 'missing'
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


# Notes, checked in order: turning points before kinds of work.
_NOTE_SCENARIOS = [
    ('found_cause', r"\b(root cause|the cause|culprit|that's why|that explains|found (it|the (bug|issue|problem)))\b"),
    ('error_seen', r'\b(error|exception|traceback|crash\w*)\b'),
    ('revert', r'\b(revert\w*|roll(ing)? back|undo\w*)\b'),
    ('retry', r'\b(retry\w*|try(ing)? again|another attempt)\b'),
    ('approval', r'\b(your approval|approve|permission|confirm with you)\b'),
    ('lint', r'\b(lint\w*|type[- ]?check\w*|phpstan|eslint|mypy|pint|prettier|tsc)\b'),
    ('plan', r'^(plan\w*|first,|the plan|here is the plan)'),
    ('refactor', r'\b(refactor\w*|clean(ing)? up|simplif\w+|extract\w*|renam\w+)\b'),
    ('security', r'\b(secret|token|password|credential|csrf|xss|sanitiz\w+)\b'),
    ('restarting', r'\b(restart\w*|relaunch\w*)\b'),
    ('database', r'\b(migrat\w+|schema|seed\w*|query|sql)\b'),
    ('ui', r'\b(css|style\w*|layout|button|modal|component|blade|view|page|ui)\b'),
    ('backend', r'\b(api|endpoint|controller|route|service|repository|backend)\b'),
]
_NOTE_RX = [(name, re.compile(rx, re.I)) for name, rx in _NOTE_SCENARIOS]
# Turning points may recur sooner than routine kinds of work.
TURNING_POINTS = {'found_cause', 'error_seen', 'revert', 'retry', 'approval', 'missing'}


def note_scenario(text):
    """Scenario named by an agent's between-steps note, if any."""
    text = (text or '').strip()
    if _MISSING.search(text):
        return 'missing'
    for name, rx in _NOTE_RX:
        if rx.search(text):
            return name
    return None


# Master's first prompt after a long break is greeted - once, across every
# session - and so is the first late-night one.
GREETING_BREAK_SECONDS = 3 * 3600
_last_prompt_at = 0.0
_late_night_on = None


def greeting(now=None):
    """'greeting_morning' / 'greeting_return' / 'late_night' for a prompt
    Master just sent, or None. Prompts cluster at 9-10am (2026-10-02 logs)."""
    global _last_prompt_at, _late_night_on
    now = now or time.time()
    hour = time.localtime(now).tm_hour
    after_break = now - _last_prompt_at >= GREETING_BREAK_SECONDS
    first_ever = _last_prompt_at == 0.0
    _last_prompt_at = now
    if after_break and not first_ever:
        return 'greeting_morning' if 5 <= hour < 12 else 'greeting_return'
    night = time.strftime('%Y-%m-%d', time.localtime(now - 6 * 3600))
    if (hour >= 22 or hour < 4) and _late_night_on != night:
        _late_night_on = night
        return 'late_night'
    return None


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
    TURNING_POINT_GAP = 30
    WORKING_GAP = 120
    LONG_TURN = 10 * 60      # 27% of turns pass this
    VERY_LONG_TURN = 30 * 60  # 6% pass this

    def __init__(self, clock=time.time):
        self.clock = clock
        self.last_phase = None
        self.last_at = 0.0
        self.phase_at = {}
        self.pending = {}  # call_id -> phase, for test runs and pushes
        self.turn_started = None
        self.long_said = set()

    def prompt(self):
        self.last_phase = None
        self.phase_at.clear()
        self.pending.clear()
        self.turn_started = self.clock()
        self.long_said.clear()

    def _turn_length(self, now):
        """'long_work' / 'very_long' once each, as the turn passes them."""
        if self.turn_started is None:
            return None
        for name, limit in (('very_long', self.VERY_LONG_TURN), ('long_work', self.LONG_TURN)):
            if now - self.turn_started >= limit and name not in self.long_said:
                self.long_said.update({name, 'long_work'})
                return name
        return None

    def note(self, scenario):
        """An agent's note named a scenario (note_scenario)."""
        if scenario is None:
            return None
        now = self.clock()
        if now - self.last_at < self.PHASE_GAP:
            return None
        gap = self.TURNING_POINT_GAP if scenario in TURNING_POINTS else self.REPEAT_GAP
        if now - self.phase_at.get(scenario, 0) < gap:
            return None
        self.last_phase = scenario
        self.phase_at[scenario] = now
        return self._emit(scenario, now)

    def tool(self, call_id, phase):
        if phase in ('test_run', 'push'):
            self.pending[call_id] = phase
        if phase == 'push':
            return None  # Announced when it succeeds.
        now = self.clock()
        if now - self.last_at >= self.PHASE_GAP:
            long_turn = self._turn_length(now)
            if long_turn:
                return self._emit(long_turn, now)
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

    def acknowledge(self):
        return self._emit('acknowledge', self.clock())

    def interrupted(self):
        self.last_phase = None
        return self._emit('interrupted', self.clock())

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
