#!/usr/bin/env python3
"""CPU-only checks for real scheduling, dependency and value ownership rules."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import unittest

spec=importlib.util.spec_from_file_location('r19_ready',Path(__file__).with_name('qwen-r19-ready.py'))
ready=importlib.util.module_from_spec(spec);spec.loader.exec_module(ready)


class ReadyTests(unittest.TestCase):
    def test_independent_cpu_runs_while_npu_owner_waits(self):
        entered=threading.Event();cpu_finished=threading.Event();main=threading.get_ident()
        def npu(values):
            self.assertNotEqual(threading.get_ident(),main)
            entered.set();self.assertTrue(cpu_finished.wait(2))
            return {'npu':values['x']+2}
        def cpu(values):
            self.assertEqual(threading.get_ident(),main)
            self.assertTrue(entered.wait(2));cpu_finished.set()
            return {'cpu':values['x']+3}
        result=ready.ReadyExecutor().run([
            ready.Operation('npu',['x'],['npu'],'npu',npu),
            ready.Operation('cpu',['x'],['cpu'],'cpu',cpu),
            ready.Operation('join',['npu','cpu'],['y'],'cpu',lambda v:{'y':v['npu']+v['cpu']})],{'x':4},['y'])
        self.assertEqual(result['values'],{'y':13})
        self.assertGreater(result['trace']['measured_cpu_npu_callback_overlap_ns'],0)
        self.assertEqual({v['value'] for v in result['trace']['lifetimes'] if v['action']=='release'},{'cpu','npu'})
    def test_prepared_program_reuses_one_persistent_owner(self):
        owners=[]
        def npu(values):owners.append(threading.get_ident());return {'y':values['x']+1}
        program=ready.PreparedProgram([ready.Operation('n',['x'],['y'],'npu',npu)],['x'],['y'])
        with ThreadPoolExecutor(max_workers=1) as owner:
            executor=ready.ReadyExecutor(owner)
            self.assertEqual(executor.run_prepared(program,{'x':1})['values']['y'],2)
            self.assertEqual(executor.run_prepared(program,{'x':4})['values']['y'],5)
        self.assertEqual(len(set(owners)),1)
    def test_external_owner_drained_on_cpu_failure(self):
        entered=threading.Event();release=threading.Event();finished=threading.Event()
        def npu(values):
            entered.set();self.assertTrue(release.wait(2));finished.set();return {'n':1}
        def cpu(values):
            self.assertTrue(entered.wait(2));release.set();raise RuntimeError('CPU failure')
        with ThreadPoolExecutor(max_workers=1) as owner:
            with self.assertRaises(RuntimeError):
                ready.ReadyExecutor(owner).run([ready.Operation('n',[],['n'],'npu',npu),ready.Operation('c',[],['c'],'cpu',cpu)],{},['n'])
            self.assertTrue(finished.is_set())
    def test_non_ssa_and_cycle_rejected(self):
        with self.assertRaises(ValueError):
            ready.ReadyExecutor().run([ready.Operation('bad',['x'],['x'],'cpu',lambda v:{'x':2})],{'x':1},['x'])
        with self.assertRaises(ValueError):
            ready.ReadyExecutor().run([ready.Operation('a',['b'],['a'],'cpu',lambda v:{'a':1}),ready.Operation('b',['a'],['b'],'cpu',lambda v:{'b':1})],{},['a'])
    def test_failed_cpu_drains_actual_owner_before_return(self):
        drained=threading.Event()
        def npu(values):drained.set();return {'n':1}
        def cpu(values):raise RuntimeError('registered CPU implementation failed')
        with self.assertRaises(RuntimeError):
            ready.ReadyExecutor().run([ready.Operation('n',[],['n'],'npu',npu),ready.Operation('c',[],['c'],'cpu',cpu)],{},['n'])
        self.assertTrue(drained.is_set())


if __name__=='__main__':unittest.main()
