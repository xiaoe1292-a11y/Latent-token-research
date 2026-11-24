"""Print a compact matrix, observed frontier and within-beam winners."""
import json
from common import ROOT, COUNTS, BEAMS, rows
from statistics_utils import pareto
from verify_results import verify


if __name__ == '__main__':
    verify(ROOT)
    data=rows(ROOT)
    output=dict(recall100_matrix={str(n):{str(b):next(r['recall@100'] for r in data if r['n']==n and r['beam']==b) for b in BEAMS} for n in COUNTS},
        winners_by_fixed_beam={str(b):max([r for r in data if r['beam']==b],key=lambda r:r['recall@100'])['point'] for b in BEAMS},
        observed_pareto=pareto(data),cost_scope='shared GPU observation, not isolated performance or service SLA')
    print(json.dumps(output,indent=2))
