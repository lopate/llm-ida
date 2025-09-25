import atexit
import faulthandler
import sys
import threading

# Enable faulthandler to write to stderr on fatal signals
faulthandler.enable(all_threads=True)

def _dump_atexit():
    try:
        handlers = getattr(atexit, '_exithandlers', None)
        if handlers is None:
            # Python 3.11+ stores handlers differently; try introspection
            handlers = []
        out = ['ATEXIT HANDLERS:']
        for h in handlers:
            try:
                out.append(repr(h))
            except Exception:
                out.append(str(h))
        sys.stderr.write('\n'.join(out) + '\n')
    except Exception as e:
        sys.stderr.write('dump atexit failed: %r\n' % (e,))

atexit.register(_dump_atexit)
"""Startup import watcher used for diagnosing native-extension segfaults.

This file is intentionally lightweight and only installed temporarily during
debugging. It wraps builtins.__import__ to record modules whose __file__ points
to a native extension (.so/.pyd/.dll). Logs are appended to /tmp/imports_watch.txt.
"""
import builtins
import sys
import atexit
import time
import faulthandler
import signal

LOG = '/tmp/imports_watch.txt'

def _append(msg: str):
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass


_orig_import = builtins.__import__

def _my_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    try:
        # try to get top-level module name
        m = sys.modules.get(name.split('.')[0], None)
        # if fromlist was used, inspect requested module
        if fromlist and name in sys.modules:
            m = sys.modules[name]
        if m is None:
            m = mod
        f = getattr(m, '__file__', None)
        if isinstance(f, str) and (f.endswith('.so') or f.endswith('.pyd') or f.endswith('.dll')):
            _append(f"{time.time():.3f} PID={os.getpid()} IMPORTED {name} -> {f}")
    except Exception:
        try:
            _append(f"{time.time():.3f} PID={os.getpid()} IMPORT ERROR inspecting {name}")
        except Exception:
            pass
    return mod


import os
builtins.__import__ = _my_import


def _dump_loaded_extensions():
    try:
        _append('--- DUMP AT EXIT ---')
        # list native extension modules
        for nm, m in list(sys.modules.items()):
            try:
                f = getattr(m, '__file__', None)
                if isinstance(f, str) and (f.endswith('.so') or f.endswith('.pyd') or f.endswith('.dll')):
                    _append(f"ATEXIT {nm} -> {f}")
            except Exception:
                # ignore per-module inspection errors
                continue

        # also log atexit handlers (best-effort)
        try:
            import atexit as _atexit
            handlers = getattr(_atexit, '_exithandlers', None)
            if handlers:
                for entry in list(handlers):
                    try:
                        fn = entry[0]
                        mod = getattr(fn, '__module__', None)
                        name = getattr(fn, '__name__', repr(fn))
                        _append(f"ATEXIT_HANDLER {name} from module {mod}")
                    except Exception:
                        _append(f"ATEXIT_HANDLER unknown {entry}")
            # Attempt to remove handlers from known native modules that have caused
            # finalizer races (best-effort, do not raise on failure).
            try:
                bad_prefixes = ('torch', 'pydantic_core', 'pandas', 'sklearn')
                new_handlers = []
                for entry in list(handlers):
                    try:
                        fn = entry[0]
                        mod = getattr(fn, '__module__', '') or ''
                        if any(mod.startswith(pref) for pref in bad_prefixes):
                            _append(f"REMOVE_ATEXIT_HANDLER {getattr(fn,'__name__',repr(fn))} from {mod}")
                            continue
                    except Exception:
                        # On error, retain the handler to be safe
                        new_handlers.append(entry)
                        continue
                    new_handlers.append(entry)
                try:
                    # replace the internal handler list
                    setattr(_atexit, '_exithandlers', new_handlers)
                except Exception:
                    # fallback: try clearing entirely if replacement fails
                    try:
                        delattr(_atexit, '_exithandlers')
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            # ignore if atexit internals not accessible
            pass
    except Exception:
        pass


atexit.register(_dump_loaded_extensions)

# Intercept atexit.register to avoid registering handlers from problematic native modules
try:
    import atexit as _atexit
    _orig_atexit_register = _atexit.register

    def _safe_register(fn, *args, **kwargs):
        try:
            mod = getattr(fn, '__module__', '') or ''
            # blacklist handlers from native torch module to avoid exit races
            if mod.startswith('torch'):
                _append(f"SKIP_ATEXIT_HANDLER {getattr(fn,'__name__',repr(fn))} from {mod}")
                return None
        except Exception:
            pass
        return _orig_atexit_register(fn, *args, **kwargs)

    _atexit.register = _safe_register
except Exception:
    # best-effort; do not fail import
    pass

# Register faulthandler handlers for fatal signals to dump Python stack traces to a file.
try:
    fh_file = open('/tmp/faulthandler_segv.log', 'w', encoding='utf-8')
    faulthandler.register(signal.SIGSEGV, file=fh_file, all_threads=True)
    faulthandler.register(signal.SIGABRT, file=fh_file, all_threads=True)
except Exception:
    try:
        # fallback: enable default faulthandler
        faulthandler.enable()
    except Exception:
        pass
