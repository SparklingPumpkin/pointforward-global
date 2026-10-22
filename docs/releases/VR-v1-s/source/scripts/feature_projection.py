"""Scene feature PCA and an alpha-aware folded first convolution.

No RGB targets are used to fit the projection. All scene projection work is
part of the measured experiment, and original payloads remain untouched.
"""
from dataclasses import replace
import json
from pathlib import Path
import time
import torch

def adapt_refiner(model, state):
    basis, mean = state['basis'], state['mean']
    old = model.refiner.stem.body[0]
    rank = basis.shape[1]
    if old.in_channels != mean.numel() + 1:
        raise ValueError('Projection and refiner feature spaces disagree')
    new = torch.nn.Conv2d(rank+1, old.out_channels, old.kernel_size,
                          padding=old.padding, bias=old.bias is not None).to(old.weight)
    with torch.no_grad():
        a, mu = basis.to(old.weight), mean.to(old.weight)
        new.weight[:,:rank].copy_(torch.einsum('oihw,ij->ojhw',old.weight[:,:-1],a))
        # Rendered centered features reconstruct as Z A^T + alpha*mean.
        new.weight[:,rank].copy_(old.weight[:,-1]+torch.einsum('oihw,i->ohw',old.weight[:,:-1],mu))
        if old.bias is not None:new.bias.copy_(old.bias)
    model.refiner.stem.body[0] = new

def project_payload(source, output, rank=32):
    source, output = Path(source), Path(output)
    output.mkdir(exist_ok=True)
    started=time.perf_counter(); samples=[]
    for name in ('static','dynamic'):
        payload=torch.load(source/f'final_{name}_128d.pt',map_location='cpu',weights_only=True)
        x=payload['colors']; samples.append(x[::max(1,len(x)//60000)][:60000].double().clone())
        del payload,x
    x=torch.cat(samples); mean=x.mean(0); centered=x-mean
    cov=centered.T@centered/len(x)
    eig,basis=torch.linalg.eigh(cov); basis=basis[:,-rank:].flip(1).float(); mean=mean.float()
    state=dict(basis=basis,mean=mean,rank=rank,source_payload=str(source),
               retained_variance=float(eig[-rank:].sum()/eig.sum()),fitting='Gaussian features only, no RGB supervision')
    for name in ('static','dynamic'):
        payload=torch.load(source/f'final_{name}_128d.pt',map_location='cpu',weights_only=True)
        features=payload['colors']; result=torch.empty((len(features),rank),dtype=features.dtype)
        for lo in range(0,len(features),250000):
            result[lo:lo+250000]=(features[lo:lo+250000]-mean)@basis
        payload['colors']=result
        torch.save(payload,output/f'final_{name}_128d.pt')
    meta=source/'global_dynamic_meta.pt'
    torch.save(torch.load(meta,map_location='cpu',weights_only=True),output/meta.name)
    state['seconds']=time.perf_counter()-started
    torch.save(state,output/'feature_projection.pt')
    (output/'feature_projection.json').write_text(json.dumps({k:v for k,v in state.items() if not torch.is_tensor(v)},indent=2))
    return state

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(4)
    print(project_payload(a.source,a.out,a.rank)['seconds'])
