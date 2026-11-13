# Copyright (c) 2019-2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>
#         Muyuan Shen <muyuan_shen@hust.edu.cn>

import atexit
import ctypes
import ctypes.util
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import weakref


SIMULATION_EARLY_ENDING = 0.5   # wait and see if the subprocess is running after creation


# ---------------------------------------------------------------------------
# Process-tree safety primitives
# ---------------------------------------------------------------------------
#
# Orphaned ns-3 processes were observed when the Python orchestrator died
# without running its `finally: exp.kill()` block (e.g. SIGKILL, lost SSH
# session, hard crash). To make cleanup robust we combine:
#
#   1) A dedicated process group / session for the waf child (so we can
#      kill the whole subtree with one killpg).
#   2) Linux PR_SET_PDEATHSIG on the waf child, which asks the kernel to
#      deliver SIGTERM to the child when its parent (this Python process)
#      dies, even via SIGKILL. wutils.py performs the same trick on the
#      ns-3 binary so the protection extends to grandchildren.
#   3) atexit + SIGINT/SIGTERM/SIGHUP handlers that invoke Experiment.kill()
#      and then re-raise the signal so the normal exit semantics survive.
#   4) A graceful SIGTERM -> wait -> SIGKILL escalation in kill_proc_tree.
#
# Together these ensure that no matter how the Python interpreter exits,
# the waf and ns-3 processes it spawned are torn down.
# ---------------------------------------------------------------------------

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>


def _set_parent_death_signal(sig=signal.SIGTERM):
    """Best-effort PR_SET_PDEATHSIG on Linux; no-op elsewhere."""
    if not sys.platform.startswith('linux'):
        return
    try:
        libc_name = ctypes.util.find_library('c') or 'libc.so.6'
        libc = ctypes.CDLL(libc_name, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception:
        # If prctl is unavailable we just lose the parent-death safety net;
        # the explicit cleanup paths still cover the common cases.
        pass


def _child_preexec():
    """preexec_fn for the waf subprocess.

    Runs in the forked child before exec. Puts the child into its own
    session/process group (so we can killpg the whole subtree) and asks
    the kernel to SIGTERM us when our parent dies.
    """
    try:
        os.setsid()
    except OSError:
        try:
            os.setpgrp()
        except OSError:
            pass
    _set_parent_death_signal(signal.SIGTERM)


def get_setting(setting_map):
    ret = ''
    for key, value in setting_map.items():
        ret += ' --{}={}'.format(key, value)
    return ret


def run_single_simulation(path, pname, setting=None, env=None, show_output=False, waf_cmd=None):
    """
    Start the ns-3 simulation via waf only: ``<waf> --run "<pname> [args]"``.

    :param path: Working directory prefix (typically './' after chdir to ns-3 root).
    :param pname: Program target name (e.g. scratch ``ntn-iab-quic-dash-ai``).
    :param waf_cmd: Override waf executable (default: env ``WAF_CMD`` or ``./waf``).
        Use e.g. ``python3 ./waf`` if ``./waf`` fails due to missing ``python``.
    """
    if env is None:
        env = {}
    env.update(os.environ)
    env['LD_LIBRARY_PATH'] = os.path.abspath(os.path.join(path, 'build', 'lib'))

    waf = waf_cmd or os.environ.get('WAF_CMD', './waf')
    # waf may be a single path (./waf) or a command with arguments
    # (python3 ./waf). Split into argv tokens. We deliberately do not
    # rewrite the path tokens: callers chdir to the ns-3 root before
    # spawning, so the relative form (./waf) resolves correctly.
    waf_argv = shlex.split(waf)
    if not waf_argv:
        raise RuntimeError('ns3ai_utils: empty waf command')

    run_arg = pname
    if setting:
        run_arg += get_setting(setting)

    argv = waf_argv + ['--run', run_arg]
    cmd_display = ' '.join(shlex.quote(a) for a in argv)

    popen_kwargs = dict(
        env=env,
        universal_newlines=True,
        stdin=subprocess.PIPE,
        preexec_fn=_child_preexec,
    )
    if not show_output:
        popen_kwargs['stdout'] = subprocess.PIPE
        popen_kwargs['stderr'] = subprocess.PIPE

    proc = subprocess.Popen(argv, **popen_kwargs)
    return cmd_display, proc


def kill_proc_tree(p, timeout=None, on_terminate=None):
    """Tear down the waf/ns-3 subtree spawned by run_single_simulation.

    Sends SIGTERM to the child's process group first, waits up to
    ``timeout`` seconds (default 3s), then escalates to SIGKILL if any
    process is still alive. Finally reaps the immediate child.
    """
    print('ns3ai_utils: Killing subprocesses...')
    _ = on_terminate  # unused; kept for signature compatibility

    pid = None
    if isinstance(p, int):
        pid = p
    elif hasattr(p, 'pid'):
        pid = p.pid
    if pid is None or pid <= 0:
        return [], []

    # Snapshot the process group ID before the child possibly exits.
    gid = None
    try:
        gid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        gid = None

    # Step 1: graceful SIGTERM to the whole group.
    if gid is not None:
        try:
            os.killpg(gid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            if hasattr(p, 'terminate'):
                p.terminate()
        except Exception:
            pass

    # Step 2: wait briefly for graceful exit.
    grace = timeout if timeout is not None else 3.0
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if hasattr(p, 'poll') and p.poll() is not None:
            break
        time.sleep(0.05)

    # Step 3: SIGKILL anything still alive in the group.
    if hasattr(p, 'poll') and p.poll() is None:
        if gid is not None:
            try:
                os.killpg(gid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            if hasattr(p, 'kill'):
                p.kill()
        except Exception:
            pass

    # Step 4: reap the immediate child to free zombies.
    if hasattr(p, 'wait'):
        try:
            p.wait(timeout=2)
        except Exception:
            pass

    return [], []


# ---------------------------------------------------------------------------
# Global cleanup / signal handling
# ---------------------------------------------------------------------------

_active_experiments = weakref.WeakSet()
_cleanup_lock = threading.Lock()
_previous_handlers = {}
_handlers_installed = False
_atexit_registered = False


def _kill_all_experiments():
    """Kill every live Experiment. Safe to call multiple times."""
    with _cleanup_lock:
        for exp in list(_active_experiments):
            try:
                exp.kill()
            except Exception:
                pass


def _install_cleanup_hooks():
    """Install atexit + SIGINT/SIGTERM/SIGHUP handlers (idempotent)."""
    global _handlers_installed, _atexit_registered

    if not _atexit_registered:
        atexit.register(_kill_all_experiments)
        _atexit_registered = True

    if _handlers_installed:
        return
    _handlers_installed = True

    def _handler(sig, frame):
        # Note: this runs in the main thread when the Python interpreter
        # next checks for pending signals. If the main thread is stuck
        # inside a C extension (e.g. the ns3-ai busy-spin) this handler
        # may be delayed - but the spawned ns-3 subtree is also covered
        # by atexit and by PR_SET_PDEATHSIG, so cleanup still happens.
        try:
            print(
                "\nns3ai_utils: signal {} received, cleaning up...".format(sig),
                flush=True,
            )
        except Exception:
            pass
        _kill_all_experiments()
        # Raise an exception so caller `finally:` blocks still run. We use
        # KeyboardInterrupt for SIGINT (matching Python's default Ctrl+C
        # semantics so existing `except KeyboardInterrupt:` clauses fire)
        # and SystemExit for SIGTERM/SIGHUP.
        if sig == signal.SIGINT:
            raise KeyboardInterrupt()
        raise SystemExit(128 + int(sig))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            prev = signal.signal(sig, _handler)
            _previous_handlers[sig] = prev
        except (OSError, ValueError):
            # Signal not available (e.g. running in a thread that can't
            # install handlers) - just skip it.
            pass


# Backwards-compatible alias retained in case anything imports the old name.
# The new global hooks are the preferred mechanism; this handler now performs
# explicit cleanup before exiting so finally-blocks in callers still execute.
def sigint_handler(sig, frame):  # pragma: no cover - legacy shim
    print("\nns3ai_utils: SIGINT detected")
    _kill_all_experiments()
    exit(1)


# This class sets up the shared memory and runs the simulation process.
class Experiment:
    _created = False

    # init ns-3 environment
    # \param[in] targetName : waf program target (e.g. ntn-iab-quic-dash-ai)
    # \param[in] waf_cmd : optional waf command (e.g. python3 ./waf); else env WAF_CMD
    def __init__(self, targetName, ns3Path, msgModule,
                 handleFinish=False,
                 useVector=False, vectorSize=None,
                 shmSize=65536,
                 segName="My Seg",
                 cpp2pyMsgName="My Cpp to Python Msg",
                 py2cppMsgName="My Python to Cpp Msg",
                 lockableName="My Lockable",
                 waf_cmd=None):
        if self._created:
            raise Exception('ns3ai_utils: Error: Experiment is singleton')
        self._created = True
        self.targetName = targetName  # waf target name, not file name
        os.chdir(ns3Path)
        self.msgModule = msgModule
        self.handleFinish = handleFinish
        self.useVector = useVector
        self.vectorSize = vectorSize
        self.shmSize = shmSize
        self.segName = segName
        self.cpp2pyMsgName = cpp2pyMsgName
        self.py2cppMsgName = py2cppMsgName
        self.lockableName = lockableName
        self.waf_cmd = waf_cmd

        self.msgInterface = msgModule.Ns3AiMsgInterfaceImpl(
            True, self.useVector, self.handleFinish,
            self.shmSize, self.segName, self.cpp2pyMsgName, self.py2cppMsgName, self.lockableName
        )
        if self.useVector:
            if self.vectorSize is None:
                raise Exception('ns3ai_utils: Error: Using vector but size is unknown')
            self.msgInterface.GetCpp2PyVector().resize(self.vectorSize)
            self.msgInterface.GetPy2CppVector().resize(self.vectorSize)

        self.proc = None
        self.simCmd = None

        # Register for global cleanup BEFORE any subprocess is spawned, so
        # even an early failure in run() still triggers teardown.
        _install_cleanup_hooks()
        _active_experiments.add(self)

        print('ns3ai_utils: Experiment initialized (waf)')

    def __del__(self):
        try:
            self.kill()
        except Exception:
            pass
        try:
            del self.msgInterface
        except Exception:
            pass
        print('ns3ai_utils: Experiment destroyed')

    # run waf --run with the setting being input
    # \param[in] setting : program argv-style parameters (default : None)
    # \param[in] show_output : whether to show output or not(default : False)
    def run(self, setting=None, show_output=False):
        self.kill()
        self.simCmd, self.proc = run_single_simulation(
            './', self.targetName, setting=setting, show_output=show_output,
            waf_cmd=self.waf_cmd)
        print("ns3ai_utils: Running simulation with: ", self.simCmd)
        # exit if an early error occurred, such as wrong target name
        time.sleep(SIMULATION_EARLY_ENDING)
        if not self.isalive():
            print('ns3ai_utils: Subprocess died very early')
            print('ns3ai_utils: Hint: build the target first: ./waf build')
            print('ns3ai_utils: Hint: if ./waf fails, try: export WAF_CMD="python3 ./waf"')
            exit(1)
        return self.msgInterface

    def kill(self):
        if self.proc is not None:
            try:
                if self.isalive():
                    kill_proc_tree(self.proc)
            except Exception:
                pass
            finally:
                self.proc = None
                self.simCmd = None

    def isalive(self):
        return self.proc is not None and self.proc.poll() is None


__all__ = ['Experiment', 'run_single_simulation']
