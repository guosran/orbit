#!/usr/bin/env python3
"""Device-free checks of both preparation boundaries and readonly storage reads."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prep_v2',HERE/'run-r19-prepared-memory-v2.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_original_frozen_main_still_has_both_guards(self):
        path=Path('/home/shiran/project/orbit-artifact/results/taskflow/qwen-whole-graph-r18/frozen-r18-v4/scripts/run-qwen-r18-runtime.py')
        before=prep.sha(path);runner=SimpleNamespace(__file__=str(path))
        prep.instrument(runner)
        self.assertEqual(before,prep.sha(path))
        self.assertIn('_r19_owned_preparation',runner.main.__code__.co_names)
        self.assertIn('prefill-loaded-original-cpu-weights',runner.main.__code__.co_consts)
        self.assertIn('decode-loaded',runner.main.__code__.co_consts)

    def test_missing_guard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runner.py'
            path.write_text('def main():\n with lock:\n  pf.load_weights(model)\n  check_memory("wrong")\n')
            with self.assertRaisesRegex(ValueError,'adjacency'):prep.instrument(SimpleNamespace(__file__=str(path)))

    def test_readonly_storage_and_two_stage_protocol(self):
        import numpy as np
        import torch
        tensor=torch.arange(8192,dtype=torch.int32);array=np.arange(8192,dtype=np.int32)
        pf=SimpleNamespace(_w={'weight':tensor,'alias':tensor[7:]},embed=array)
        decoder=SimpleNamespace(Wv16=array)
        original_tensor=tensor.clone();original_array=array.copy()
        ranges,owners=prep.weight_ranges(pf,None)
        self.assertEqual(sum(hi-lo for lo,hi,_ in ranges),tensor.untyped_storage().nbytes()+array.nbytes)
        with tempfile.TemporaryDirectory() as directory:
            for stage,dec in [('prefill-loaded-original-cpu-weights',None),('decode-loaded',decoder)]:
                prep.release_and_prefault(pf,dec,directory,{'test':True},stage)
            protocol=json.loads((Path(directory)/'preparation-protocol.json').read_text())
            self.assertEqual(protocol['mode'],'owned-prefill-decode-release-readonly-prefault-v2')
            self.assertEqual([r['stage'] for r in protocol['proofs']],['prefill-loaded-original-cpu-weights','decode-loaded'])
            for record in protocol['proofs']:
                self.assertEqual(record['sha256'],prep.sha(record['file']))
                proof=json.loads(Path(record['file']).read_text())
                self.assertTrue(proof['reads_only']);self.assertEqual(proof['bo_writes'],0)
                self.assertGreater(proof['pages_touched'],0)
        self.assertTrue(torch.equal(tensor,original_tensor));self.assertTrue(np.array_equal(array,original_array))


if __name__=='__main__':unittest.main()
