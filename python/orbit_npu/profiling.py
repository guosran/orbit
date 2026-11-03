"""Add a read-only timer to a locally generated XRT ELF launcher.

No installed files are modified. The generated source is part of Triton's
native launcher cache key. Exact source guards fail closed if the pinned
wheel's template changes. Times are host steady-clock intervals, not device
cycle counters.
"""
def install():
    from triton.backends.amd_triton_npu import driver
    if getattr(driver, '_orbit_profile_installed', False):
        return
    original = driver._generate_elf_launcher

    def generate(*args, **kwargs):
        source = original(*args, **kwargs)
        edits = [
            ('static char elf_path[1024]',
             'static double orbit_load_seconds = 0;\nstatic char elf_path[1024]'),
            ('if (!s.ready) {',
             'if (!s.ready) {\n        auto orbit_load_start = std::chrono::steady_clock::now();'),
            ('s.ready = true;',
             's.ready = true;\n        orbit_load_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - orbit_load_start).count();'),
            ('static PyMethodDef ModuleMethods[] = {',
             'static PyObject* orbit_profile(PyObject*, PyObject*) { return PyFloat_FromDouble(orbit_load_seconds); }\nstatic PyMethodDef ModuleMethods[] = {\n  {"orbit_load_seconds", orbit_profile, METH_NOARGS, "Cumulative device/ELF/context/kernel load host seconds"},'),
        ]
        for old, new in edits:
            if source.count(old) != 1:
                raise RuntimeError('Unsupported XRT launcher version: instrumentation guard: '+old)
            source = source.replace(old,new)
        return source
    # aircc otherwise writes its default air_project/ and params.txt in the
    # caller's cwd, even when the ELF path is absolute. Change only the CHILD
    # compiler cwd and pass an explicit temporary directory; never os.chdir.
    original_compile = driver._run_compile
    def compile_in_output(cmd, env=None):
        from pathlib import Path
        import subprocess, sys, json
        from triton.backends.amd_triton_npu.config import npu_config
        if Path(str(cmd[0])).name not in ('aircc','aircc.exe'):
            return original_compile(cmd,env=env)
        build = Path(npu_config.air_project_path).resolve()
        build.mkdir(parents=True,exist_ok=True)
        command = list(cmd)
        command.insert(-1,'--tmpdir='+str(build))
        with (build.parent/'compile-commands.jsonl').open('a') as out:
            out.write(json.dumps(dict(command=command,cwd=str(build.parent)))+'\n')
        result = subprocess.run(command,cwd=build.parent,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (build.parent/'aircc-build.log').write_bytes(result.stdout)
        if result.returncode:
            sys.stderr.write(result.stdout.decode('utf-8',errors='replace'))
            raise subprocess.CalledProcessError(result.returncode,command,output=result.stdout)
    # NPUChain uses a Python XRT runner instead of the generated C launcher.
    # Time only opening an already compiled ELF. The wrapper also observes
    # later LRU reopens, without preloading contexts or changing dispatches.
    import inspect, time
    from triton.backends.amd_triton_npu.multilaunch import NPUChain
    original_open = NPUChain._open_runner
    open_source = inspect.getsource(original_open)
    for marker in ('if self._elf_path is None:', 'self._runner = runner(self._elf_path, self._kernel_name)'):
        if marker not in open_source:
            raise RuntimeError('Unsupported NPUChain loading timer guard: '+marker)
    driver._orbit_chain_load_seconds = 0.0
    def open_profiled(chain):
        if chain._elf_path is None:
            raise RuntimeError('ORBIT load timing requires explicitly compiled NPUChain ELF')
        start = time.perf_counter()
        try:
            return original_open(chain)
        finally:
            driver._orbit_chain_load_seconds += time.perf_counter()-start
    NPUChain._open_runner = open_profiled
    driver._run_compile = compile_in_output
    driver._generate_elf_launcher = generate
    driver._orbit_profile_installed = True


def load_breakdown():
    from triton.backends.amd_triton_npu import driver
    # Multiple argument specializations may refer to one native module.
    modules = {id(m):m for m in driver._global_module_cache.values()}
    return dict(staged_load_seconds=sum(m.orbit_load_seconds() for m in modules.values() if hasattr(m,'orbit_load_seconds')),
                chain_load_seconds=getattr(driver,'_orbit_chain_load_seconds',0.0))


def load_seconds():
    return sum(load_breakdown().values())
