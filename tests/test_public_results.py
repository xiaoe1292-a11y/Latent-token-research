"""Corrupt actual published data to prove the numerical audit rejects it."""
import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT
from verify_results import verify


class ActualEvidenceControls(unittest.TestCase):
    def corrupted_csv(self, column, transform):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'data',root/'data')
            shutil.copytree(ROOT/'configs',root/'configs')
            path=root/'data/joint16/results.csv'
            with path.open(encoding='utf-8',newline='') as stream:
                rows=list(csv.DictReader(stream))
            rows[0][column]=str(transform(float(rows[0][column])))
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Recall@100' if column=='recall@100' else 'memory MiB'):
                verify(root)

    def test_changed_recall_is_rejected_against_real_rank_arrays(self):
        self.corrupted_csv('recall@100',lambda value:value+.001)

    def test_wrong_memory_units_are_rejected_against_raw_bytes(self):
        self.corrupted_csv('peak_allocated_mib',lambda value:value*1048576)

    def test_reclassifying_shared_measurements_as_exclusive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'data',root/'data')
            shutil.copytree(ROOT/'configs',root/'configs')
            import json
            path=root/'data/joint16/profiles.json'
            profiles=json.loads(path.read_text())
            profiles['n02-b0250']['exclusive_gpu']=True
            path.write_text(json.dumps(profiles),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'cost scope must remain shared'):
                verify(root)


if __name__=='__main__':
    unittest.main()
