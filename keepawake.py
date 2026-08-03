"""Hold a wake lock for the duration of a long run.

Windows 11 on this machine uses Modern Standby (S0ix), which suspends running processes
rather than merely idling the CPU, so a training job dies when the machine sleeps. Rather
than change machine-wide power settings and have to remember to undo them, a process can
ask Windows to keep the system awake for as long as it runs.

SetThreadExecutionState with ES_CONTINUOUS holds the request until the process clears it
or exits, so the lock cannot outlive the job.

Note ES_DISPLAY_REQUIRED is deliberately NOT set: the screen is allowed to switch off, only
the system is kept from sleeping. Closing the lid is still governed by the lid action, so if
the lid action is "sleep" this alone will not save a run. See README for that case.

Usage:
    with keep_awake():
        ...long job...
"""

import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


@contextlib.contextmanager
def keep_awake(label="job"):
    if not sys.platform.startswith("win"):
        yield
        return

    import ctypes

    kernel32 = ctypes.windll.kernel32
    # away mode keeps work running with the screen off; falls back cleanly if unsupported
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    ok = kernel32.SetThreadExecutionState(flags)
    if not ok:
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ok = kernel32.SetThreadExecutionState(flags)
    print(f"[keepawake] sleep {'blocked' if ok else 'NOT blocked (call failed)'} for {label}",
          flush=True)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[keepawake] released", flush=True)
