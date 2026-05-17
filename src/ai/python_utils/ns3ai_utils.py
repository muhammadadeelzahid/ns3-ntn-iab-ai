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

import os
import shlex
import subprocess
import time
import signal


SIMULATION_EARLY_ENDING = 0.5   # wait and see if the subprocess is running after creation


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
    # Single-path waf (e.g. ./waf) is resolved under path; commands with spaces
    # (e.g. python3 ./waf) are used as-is for shell=True.
    if (not os.path.isabs(waf)) and (' ' not in waf):
        waf = os.path.join(path, waf)
        waf = os.path.normpath(waf)
    run_arg = pname
    if setting:
        run_arg += get_setting(setting)
    cmd = '{} --run {}'.format(waf, shlex.quote(run_arg))

    if show_output:
        proc = subprocess.Popen(cmd, shell=True, universal_newlines=True, env=env,
                                  stdin=subprocess.PIPE,
                                  preexec_fn=os.setpgrp)
    else:
        proc = subprocess.Popen(cmd, shell=True, universal_newlines=True, env=env,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                preexec_fn=os.setpgrp)

    return cmd, proc


# used to kill the waf/simulation process and its child processes
def kill_proc_tree(p, timeout=None, on_terminate=None):
    print('ns3ai_utils: Killing subprocesses...')
    # waf is started with setpgrp — kill the whole process group.
    _ = on_terminate  # unused; kept for signature compatibility
    pid = None
    if isinstance(p, int):
        pid = p
    elif hasattr(p, 'pid'):
        pid = p.pid
    if pid is not None and pid > 0:
        try:
            gid = os.getpgid(pid)
            os.killpg(gid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                if hasattr(p, 'kill'):
                    p.kill()
            except Exception:
                pass
    if hasattr(p, 'wait'):
        wait_s = timeout if timeout is not None else 5
        try:
            p.wait(timeout=wait_s)
        except Exception:
            pass
    return [], []


# According to Python signal docs, after a signal is received, the
# low-level signal handler sets a flag which tells the virtual machine
# to execute the corresponding Python signal handler at a later point.
#
# As a result, a long ns-3 simulation, during which no C++-Python
# interaction occurs (such as the Multi-BSS example), may run uninterrupted
# for a long time regardless of any signals received.
def sigint_handler(sig, frame):
    print("\nns3ai_utils: SIGINT detected")
    exit(1)  # this will execute the `finally` block


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
        print('ns3ai_utils: Experiment initialized (waf)')

    def __del__(self):
        self.kill()
        del self.msgInterface
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
        signal.signal(signal.SIGINT, sigint_handler)
        return self.msgInterface

    def kill(self):
        if self.proc and self.isalive():
            kill_proc_tree(self.proc)
            self.proc = None
            self.simCmd = None

    def isalive(self):
        return self.proc.poll() is None


__all__ = ['Experiment', 'run_single_simulation']
