#!/usr/bin/env python3
import json, pathlib, tempfile, hashlib, subprocess, unittest, sys
import importlib.util
SCRIPT = str(pathlib.Path(__file__).resolve().parents[2] / 'scripts/build-heterogeneous-calibration-index.py')

def save(p, d):
    p.write_text(json.dumps(d))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

class IntakeTest(unittest.TestCase):

    def test_seals_without_reading_labels_and_rejects_foreign_graph(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            cases = []
            for split, n in [('train', 16), ('holdout', 48)]:
                m = root / split
                m.mkdir()
                manifest = m / 'manifest.json'
                compiled = m / 'compiled.json'
                identity = m / 'identity.json'
                features = m / 'features.json'
                facts = m / 'mechanism-facts.json'
                save(manifest, {'id': split, 'graph_id': split, 'execution_contract_identity': 'synthetic-contract-no-hardware', 'regions': [{'index': 0, 'device': 'cpu', 'child_graph_id': split + '-child'}]})
                save(compiled, {'manifest': str(manifest), 'manifest_sha256': sha(manifest), 'runtime_identity': 'orbit.serial-prepared-typed-native-wait2.v32', 'plan_id': split, 'regions': [{'index': 0, 'device': 'cpu', 'child_graph_id': split + '-child'}]})
                save(identity, {'compiled': str(compiled), 'compiled_sha256': sha(compiled), 'runtime': 'orbit.serial-prepared-typed-native-wait2.v32', 'plan_id': split})
                save(features, {'schema': 'orbit.heterogeneous.edge-features.v32', 'parent_graph_id': split, 'application': {'call': 1, 'input_bytes': 2, 'output_bytes': 4, 'configuration_count': 0}, 'regions': [{'region': 0, 'device': 'cpu', 'child_graph_id': split + '-child', 'features': {'call': 1, 'source_operations': 12}}]})
                save(facts, {'schema': 'orbit.mechanism-facts.v13', 'program_graph_id': split, 'items': [{'mechanism': 'host.input_reset', 'bytes': 2}, {'mechanism': 'host.output_copy', 'bytes': 4}]})
                (m / 'samples.json').write_text('NOT JSON: confidential timing labels are not parsed by index intake')
                save(m / 'result.json', {'synthetic': True})
                save(m / 'process.json', {'pid_scope': 'measurement-python', 'identity_sha256': sha(identity)})
                cases.append({'id': split, 'shape': n, 'split': split, 'ranking_group': split, 'measurement': str(m), 'features': str(features), 'mechanism_facts': str(facts)})
            spec = root / 'spec.json'
            reg = root / 'registration.json'
            save(spec, {'cases': cases})
            subprocess.run([sys.executable, SCRIPT, 'preregister', '--spec', str(spec), '--output', str(reg)], check=True, capture_output=True)
            subprocess.run([sys.executable, SCRIPT, 'build', '--spec', str(spec), '--registration', str(reg), '--output', str(root / 'good')], check=True, capture_output=True)
            self.assertEqual(len(json.loads((root / 'good/index.json').read_text())['cases']), 2)
            f = pathlib.Path(cases[1]['features'])
            d = json.loads(f.read_text())
            d['regions'][0]['child_graph_id'] = 'foreign-child'
            save(f, d)
            p = subprocess.run([sys.executable, SCRIPT, 'build', '--spec', str(spec), '--registration', str(reg), '--output', str(root / 'bad')], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('exact ordered child witness mismatch', p.stderr)
class FeatureConfigTest(unittest.TestCase):

    def setUp(self):
        path = pathlib.Path(SCRIPT).with_name('make-heterogeneous-feature-config.py')
        spec = importlib.util.spec_from_file_location('feature_config', path)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.script = path

    def tiles(self, configuration):
        dimensions = configuration['program_search']['dimensions']
        return [next(d for d in dimensions if d['id'] == 'tiling.' + axis)
                ['choices'][0]['tile'] for axis in ('m', 'n', 'k')]

    def test_independent_tiles_preserve_axis_order(self):
        for tiles in ((16, 16, 64), (16, 32, 48), (16, 16, 48)):
            config = self.helper.configuration({}, 2, 4, 'npu', tiles, True)
            self.assertEqual(self.tiles(config), list(tiles))
            self.assertTrue(all(len(d['choices']) == 1
                                for d in config['program_search']['dimensions']))

    def test_legacy_uniform_tile_is_compatible(self):
        uniform = self.helper.configuration({}, 2, 4, 'cpu', 32, True)
        explicit = self.helper.configuration({}, 2, 4, 'cpu', (32, 32, 32), True)
        self.assertEqual(uniform, explicit)

    def test_invalid_tile_extents_are_rejected(self):
        for tiles in ((16, 16), (16, 0, 48), (16, -1, 48), (16, True, 48)):
            with self.assertRaisesRegex(ValueError, 'three positive integer'):
                self.helper.configuration({}, 2, 4, 'npu', tiles, True)

    def test_cli_accepts_triple_and_rejects_conflicting_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            base, output = root / 'base.json', root / 'config.json'
            base.write_text('{}')
            command = [sys.executable, str(self.script), '--base', str(base),
                       '--output', str(output), '--tasks', '2', '--storages', '4',
                       '--device', 'npu', '--tiles', '16', '32', '48']
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(self.tiles(json.loads(output.read_text())), [16, 32, 48])
            conflict = subprocess.run(command + ['--tile', '16'], capture_output=True)
            self.assertNotEqual(conflict.returncode, 0)


if __name__ == '__main__':
    unittest.main()
