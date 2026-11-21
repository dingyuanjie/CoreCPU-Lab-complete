import pytest
import torch
from corecpu_lab.models.factory import build_model,ARCHITECTURES
from corecpu_lab.tasks.structured.dataset import generator,make_batch,only_inputs
from corecpu_lab.training.losses import structured_loss,language_loss

@pytest.mark.parametrize('name',list(ARCHITECTURES))
def test_structured_four_models(name,small_config):
    org,compute=ARCHITECTURES[name]
    cfg=small_config(org,compute)
    model=build_model(cfg)
    batch=make_batch(4,3,generator(200),'train')
    assert 'labels' not in only_inputs(batch) and 'qpos' not in only_inputs(batch)
    out=model(**only_inputs(batch))
    assert out.logits.shape==(4,3,8)
    assert out.attention_logits.shape==(4,4,8)
    assert out.finite()
    loss,_=structured_loss(out,batch,cfg)
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()

@pytest.mark.parametrize('name',list(ARCHITECTURES))
def test_language_four_models(name,small_config):
    org,compute=ARCHITECTURES[name]
    cfg=small_config(org,compute,'language')
    model=build_model(cfg)
    x=torch.randint(0,256,(2,8))
    out=model(x)
    assert out.logits.shape==(2,8,257)
    loss,_=language_loss(out,{'labels':x},cfg)
    loss.backward()
    assert model.token.weight.grad is not None


def test_sparse_skips_unselected_experts(small_config):
    cfg=small_config('corecpu','conventional')
    m=build_model(cfg)
    with torch.no_grad():
        m.runtime.router.weight.zero_()
        m.runtime.router.bias.copy_(torch.tensor([10.0,-10.0,-10.0]))
    b=make_batch(3,2,generator(111),'iid')
    loss=structured_loss(m(**only_inputs(b)),b,cfg)[0]
    loss.backward()
    assert m.runtime.experts[0].fn[0].weight.grad is not None
    assert all(p.grad is None for e in m.runtime.experts[1:] for p in e.parameters())

@pytest.mark.parametrize('name',list(ARCHITECTURES))
def test_language_causality(name,small_config):
    org,compute=ARCHITECTURES[name]
    cfg=small_config(org,compute,'language')
    m=build_model(cfg).eval()
    x=torch.randint(0,256,(2,8))
    altered=x.clone();altered[:,-1]=(altered[:,-1]+1)%256
    with torch.no_grad():
        original=m(x).logits[:,:-1]
        changed=m(altered).logits[:,:-1]
    torch.testing.assert_close(original,changed,atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize('org',('sequential','corecpu'))
def test_sec_streamed_equals_full_sequence(org,small_config):
    cfg=small_config(org,'sec','language')
    m=build_model(cfg).eval()
    x=torch.randint(0,256,(2,8))
    with torch.no_grad():
        full=m(x).logits
        first=m(x[:,:4])
        second=m(x[:,4:],state=first.state)
        merged=torch.cat((first.logits,second.logits),dim=1)
    torch.testing.assert_close(full,merged,rtol=1e-5,atol=1e-5)
