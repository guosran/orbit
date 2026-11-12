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
class ExplicitDesignPolicyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.source = self.root / 'source.mlir'
        self.source.write_text('module {} // synthetic source identity only\n')
        self.cases = []
        for split in ('train', 'holdout'):
            folder = self.root / split
            folder.mkdir()
            features = folder / 'features.json'
            save(features, {
                'schema': 'orbit.heterogeneous.edge-features.v32',
                'parent_graph_id': 'same-parent-source',
                'application': {'call': 1, 'input_bytes': 2, 'output_bytes': 4,
                                'configuration_count': 0},
                'regions': [{'region': 0, 'device': 'cpu',
                             'child_graph_id': split + '-child',
                             'features': {'call': 1, 'source_operations': 12}}],
            })
            self.cases.append({
                'id': split, 'split': split, 'shape': 7,
                'ranking_group': 'same-source-input', 'measurement': str(folder),
                'features': str(features), 'mechanism_facts': str(folder / 'facts.json'),
                'design': {'source': {'path': str(self.source), 'sha256': sha(self.source)},
                           'parent_graph_id': 'same-parent-source',
                           'region_graph_ids': [split + '-child']},
            })

    def invoke(self, phase='preregister', extra_cases=None, policy='explicit-case-design-v1'):
        spec = self.root / 'spec.json'
        save(spec, {'split_policy': policy,
                    'cases': self.cases if extra_cases is None else extra_cases})
        command = [sys.executable, SCRIPT, phase, '--spec', str(spec),
                   '--output', str(self.root / ('registration.json' if phase == 'preregister' else 'index'))]
        if phase == 'build':
            command += ['--registration', str(self.root / 'registration.json')]
        return subprocess.run(command, capture_output=True, text=True)

    def test_same_shape_distinct_children_and_repeated_processes_are_valid(self):
        duplicate = json.loads(json.dumps(self.cases[0]))
        duplicate['id'] = 'another-fresh-process'
        result = self.invoke(extra_cases=self.cases + [duplicate])
        self.assertEqual(result.returncode, 0, result.stderr)
        registration = json.loads((self.root / 'registration.json').read_text())
        designs = registration['case_designs']
        self.assertEqual(designs['train']['design_sha256'],
                         designs['another-fresh-process']['design_sha256'])
        self.assertNotEqual(designs['train']['design_sha256'], designs['holdout']['design_sha256'])
        self.assertEqual(registration['split_policy'], 'explicit-case-design-v1')

    def test_case_ids_shape_and_source_alias_cannot_hide_cross_split_design(self):
        feature = pathlib.Path(self.cases[1]['features'])
        data = json.loads(feature.read_text())
        data['regions'][0]['child_graph_id'] = 'train-child'
        save(feature, data)
        self.cases[1]['id'] = 'renamed-holdout'
        self.cases[1]['shape'] = 9
        self.cases[1]['design']['region_graph_ids'] = ['train-child']
        # Even distinct source bytes cannot turn the same typed witness into a
        # heldout execution design: the authoritative graph IDs are identical.
        alias = self.root / 'source-comment-alias.mlir'
        alias.write_text(self.source.read_text() + '// another comment\n')
        self.cases[1]['design']['source'] = {'path': str(alias), 'sha256': sha(alias)}
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('same typed design appears across', result.stderr)

    def test_design_must_match_native_feature_witness(self):
        self.cases[1]['design']['region_graph_ids'] = ['foreign-child']
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differs from native feature witness', result.stderr)

    def test_explicit_shape_requires_positive_integer_and_real_source_hash(self):
        for value in (0, -1, True, 7.0):
            with self.subTest(shape=value):
                self.cases[0]['shape'] = value
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('positive integer', result.stderr)
        self.cases[0]['shape'] = 7
        self.cases[0]['design']['source']['sha256'] = '0' * 64
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source SHA mismatch', result.stderr)

    def test_legacy_default_and_unknown_policy_still_reject(self):
        self.assertIn('shape outside', self.invoke(policy='legacy-shape-split-v32').stderr)
        self.assertIn('unknown split policy', self.invoke(policy='implicit-random-split').stderr)
        del self.cases[0]['design']
        self.assertIn('complete typed witness', self.invoke().stderr)

    def test_build_preserves_exact_compiled_checks_and_freezes_designs(self):
        for case in self.cases:
            folder = pathlib.Path(case['measurement'])
            child = case['design']['region_graph_ids'][0]
            manifest = folder / 'manifest.json'
            save(manifest, {'id': case['id'], 'graph_id': 'same-parent-source',
                            'execution_contract_identity': 'synthetic-no-device',
                            'regions': [{'index': 0, 'device': 'cpu', 'child_graph_id': child}]})
            compiled = folder / 'compiled.json'
            save(compiled, {'manifest': str(manifest), 'manifest_sha256': sha(manifest),
                            'runtime_identity': 'orbit.serial-prepared-typed-native-wait2.v32',
                            'plan_id': case['id'],
                            'regions': [{'index': 0, 'device': 'cpu', 'child_graph_id': child}]})
            identity = folder / 'identity.json'
            save(identity, {'compiled': str(compiled), 'compiled_sha256': sha(compiled),
                            'runtime': 'orbit.serial-prepared-typed-native-wait2.v32',
                            'plan_id': case['id']})
            save(pathlib.Path(case['mechanism_facts']), {
                'schema': 'orbit.mechanism-facts.v13', 'program_graph_id': 'same-parent-source',
                'items': [{'mechanism': 'host.input_reset', 'bytes': 2},
                          {'mechanism': 'host.output_copy', 'bytes': 4}]})
            (folder / 'samples.json').write_text('private timing bytes are not parsed')
            save(folder / 'result.json', {'synthetic': True})
            save(folder / 'process.json', {'pid_scope': 'measurement-python',
                                          'identity_sha256': sha(identity)})
        registered = self.invoke()
        self.assertEqual(registered.returncode, 0, registered.stderr)
        built = self.invoke(phase='build')
        self.assertEqual(built.returncode, 0, built.stderr)
        index = json.loads((self.root / 'index/index.json').read_text())
        self.assertEqual(index['split_policy'], 'explicit-case-design-v1')
        self.assertEqual(len(index['case_designs']), 2)
        self.assertTrue(all(row['design_sha256'] for row in index['cases']))
        # Frozen static evidence is rechecked before measurement labels open.
        self.source.write_text(self.source.read_text() + '// changed\n')
        changed = self.invoke(phase='build')
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn('source SHA mismatch', changed.stderr)


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
