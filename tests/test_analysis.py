from pathlib import Path
from corecpu_lab.benchmarks.analysis import analyze_matrix


def test_factorial_effects(tmp_path):
    p=tmp_path/'results.csv'
    p.write_text('architecture,seed,split,accuracy\n'
        'A,1,iid,0.5\nB,1,iid,0.6\nC,1,iid,0.7\nD,1,iid,0.9\n'
        'A,2,iid,0.4\nB,2,iid,0.5\nC,2,iid,0.6\nD,2,iid,0.8\n')
    r=analyze_matrix(p,tmp_path/'analysis.json')
    assert r['splits']['iid']['paired_seeds']==[1,2]
    assert abs(r['splits']['iid']['effects']['interaction']['mean']-0.1)<1e-9
    assert (tmp_path/'analysis.json').exists()
