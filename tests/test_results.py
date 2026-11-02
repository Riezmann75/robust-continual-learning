import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ResultProvenanceTests(unittest.TestCase):
    def test_historical_sources_and_evaluation_sizes(self):
        entries=json.loads((ROOT/'results/manifest.json').read_text())
        self.assertEqual(len(entries),11)
        for item in entries:
            evaluation=json.loads((ROOT/'results'/item['evaluation']).read_text())
            provenance=json.loads((ROOT/'results'/item['provenance']).read_text())
            folder=(ROOT/'results'/item['provenance']).parent
            self.assertEqual(evaluation['clean']['n'],10000)
            self.assertEqual(evaluation['corrupted_examples'],750000)
            self.assertEqual(len(evaluation['corruptions']),15)
            for name,digest in provenance['source_hashes'].items():
                self.assertEqual(hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest(),digest)
    def test_new_configs_are_explicit(self):
        for name,decoder in [('augce',False),('augce_decoder',True)]:
            cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
            self.assertEqual(cfg['args']['epochs_per_task'],10)
            self.assertEqual(cfg['args']['decoder'],decoder)
            self.assertEqual(cfg['args']['lambda_aug'],1.)
