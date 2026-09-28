"""E043: frozen E038 CNO, persist-first-4, Fold-B-fitted interval bounds."""
import json
import os
import sys
import numpy as np
import torch

HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0,HERE)
with open(os.path.join(HERE,"calibration.json")) as stream:
    CAL=json.load(stream)
STATE={}


def conditional_bounds(point, inputs):
    # Per-window feature: never infer a threshold from the evaluation batch.
    feature=np.std(inputs[...,:2],axis=1).mean(axis=(1,2,3))
    bins=(feature>CAL["threshold"]).astype(int)
    half=np.float32(.025)*np.abs(point)+np.float32(.15*CAL["sigma_global"])
    scales=np.asarray(CAL["scales"],dtype=np.float32)
    for h,(a,b) in enumerate(CAL["horizons"]):
        for c in range(2):
            half[:,a:b,:,:,c]*=scales[bins,h,c][:,None,None,None]
    half[...,2]=0
    lower=point-half; upper=point+half
    lower[...,2]=upper[...,2]=0
    return lower,upper


def predict(input_array,metadata=None):
    del metadata
    value=np.asarray(input_array,dtype=np.float32)
    if value.ndim!=5 or value.shape[1:]!=(20,32,64,3) or len(value)==0 or not np.isfinite(value).all():
        raise ValueError("expected nonempty finite N,20,32,64,3 input")
    if not STATE:
        from load_baseline import build_model
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=build_model("cno",device=str(device))
        checkpoint=torch.load(os.path.join(HERE,"model.pth"),map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model.eval()
        STATE.update(model=model,device=device,normalizer=checkpoint["normalizer"])
    device=STATE["device"]; stats=STATE["normalizer"]
    mi=torch.tensor(stats["mean_input"],device=device,dtype=torch.float32)
    si=torch.tensor(stats["std_input"],device=device,dtype=torch.float32)
    mt=torch.tensor(stats["mean_target"],device=device,dtype=torch.float32)
    st=torch.tensor(stats["std_target"],device=device,dtype=torch.float32)
    output=np.empty_like(value)
    with torch.inference_mode():
        for i in range(len(value)):
            tensor=torch.from_numpy(value[i:i+1]).to(device)
            pred=STATE["model"]((tensor-mi)/si)*st+mt
            output[i:i+1]=pred.cpu().numpy()
    output[:,:4]=value[:,-1:]
    output[...,2]=0
    lower,upper=conditional_bounds(output,value)
    if not all(np.isfinite(a).all() for a in (output,lower,upper)):
        raise FloatingPointError("nonfinite forecast or interval")
    return {"prediction":output,"lower":lower,"upper":upper}
