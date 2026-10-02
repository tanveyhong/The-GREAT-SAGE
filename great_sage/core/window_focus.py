"""Bring a coding agent's app to the front (the overlay's activity rows).

The Claude desktop app runs every session in one window, so the target is
the app, not the session; Codex is a separate app. Windows refuses
SetForegroundWindow to a process that is not in the foreground already -
a press-and-release of Alt first is the standard way round that.
"""
import ctypes
import ctypes.wintypes as wt
import logging

log = logging.getLogger(__name__)

# Executable names per agent, lower-case.
APPS = {
    'claude': ('claude.exe',),
    'codex': ('codex.exe', 'chatgpt.exe'),
}


def _exe_name(pid):
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ''
    try:
        size = wt.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.rsplit('\\', 1)[-1].lower()
        return ''
    finally:
        kernel.CloseHandle(handle)


def find_window(agent):
    """The largest visible, titled top-level window of the agent's app."""
    names = APPS.get(str(agent).lower())
    if not names:
        return None
    user = ctypes.windll.user32
    best, best_area = None, 0

    def visit(hwnd, _):
        nonlocal best, best_area
        if not user.IsWindowVisible(hwnd) or user.GetWindowTextLengthW(hwnd) == 0:
            return True
        pid = wt.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if _exe_name(pid.value) not in names:
            return True
        rect = wt.RECT()
        user.GetWindowRect(hwnd, ctypes.byref(rect))
        area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
        if area > best_area:
            best, best_area = hwnd, area
        return True

    user.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(visit), 0)
    return best


def focus_agent(agent):
    """Restore and raise the agent's window. True if one was found."""
    hwnd = find_window(agent)
    if not hwnd:
        log.info('No %s window to bring forward', agent)
        return False
    user = ctypes.windll.user32
    if user.IsIconic(hwnd):
        user.ShowWindow(hwnd, 9)          # SW_RESTORE
    user.keybd_event(0x12, 0, 0, 0)       # Alt down/up: lifts the foreground lock
    user.keybd_event(0x12, 0, 2, 0)
    user.SetForegroundWindow(hwnd)
    return True
