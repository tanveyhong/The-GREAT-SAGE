# PyInstaller spec for the Coding Agent Companion build.
#
#   py build.py --companion
#
# The Companion (APP_MODE = "companion", the default) plays pre-made clips
# and never loads a TTS model, translator or speech recogniser, so it needs
# none of torch, TTS, F5, faster-whisper or transformers - the bulk of the
# full ~5.5GB bundle. They are excluded outright: a module that tries them
# lazily (push-to-talk, the full voice engines) reports itself unavailable
# instead of shipping gigabytes for a mode this build is not.
#
# The overlay is built separately (overlay.spec, Python 3.11) and merged in
# by build.py, exactly as for the full build. voice_lines/ carries the
# scenario clips from voice_lines/agent - generate them before building
# (py -m great_sage.voice.make_agent_clips); build.py refuses without them.

import os

block_cipher = None

datas = [
    ('hud_prototype.html', '.'),
    ('installer.html', '.'),
    ('log_console.html', '.'),
    ('voice_lines', 'voice_lines'),   # recorded lines + generated scenario clips
    ('vendor', 'vendor'),             # three.js, so the HUD renders offline
    ('assets', 'assets'),             # interface sounds
    ('defaults', 'defaults'),         # shipped settings
]

hiddenimports = [
    # pywebview picks its GUI backend at runtime.
    'webview.platforms.winforms',
    'clr_loader',
    'pythonnet',
    # The clip voice: effects chain and file playback.
    'pedalboard',
    'soundfile',
    'sounddevice',
    'websockets',
]

excludes = [
    # The live voice, translation and speech recognition stacks.
    'torch', 'torchaudio', 'torchvision', 'torchcodec',
    'TTS', 'coqui_tts', 'f5_tts', 'vocos', 'x_transformers', 'transformers',
    'faster_whisper', 'ctranslate2', 'onnxruntime', 'pocket_tts', 'pyttsx3',
    'cutlet', 'fugashi', 'unidic_lite', 'librosa', 'numba', 'llvmlite',
    'datasets', 'pyarrow', 'pandas', 'matplotlib', 'accelerate', 'safetensors',
    'bitsandbytes', 'hydra', 'omegaconf', 'cached_path',
    # The overlay lives in its own bundle.
    'PySide6', 'shiboken6', 'qtpy',
    # Dev-only.
    'PyInstaller', 'pytest', 'IPython', 'jupyter', 'notebook', 'tkinter',
]

a = Analysis(
    ['app.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='GreatSage',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon='icon.ico',
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name='GreatSage',
)
