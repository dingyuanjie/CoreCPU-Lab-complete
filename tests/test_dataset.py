import torch
from corecpu_lab.tasks.structured.dataset import generator,make_batch,only_inputs


def test_batch_reproducible():
    a=make_batch(8,3,generator(313),'train')
    b=make_batch(8,3,generator(313),'train')
    for k in a:
        torch.testing.assert_close(a[k],b[k])


def test_model_never_receives_targets():
    batch=make_batch(4,3,generator(7),'iid')
    inp=only_inputs(batch)
    assert set(inp)=={'keys','values','queries','use_ref','literals','ops','mod'}


def test_split_pair_and_new_moduli():
    pair=make_batch(128,3,generator(71),'pair')
    p=pair['ops']
    held=((p[:,:-1]==0)&(p[:,1:]==2))|((p[:,:-1]==1)&(p[:,1:]==3))
    assert bool(held.any(-1).all())
    novel=make_batch(128,3,generator(72),'mod')
    assert set(novel['mod'].tolist()).issubset({5,7})
    train=make_batch(128,3,generator(73),'train')
    assert set(train['mod'].tolist()).issubset({4,6,8})
