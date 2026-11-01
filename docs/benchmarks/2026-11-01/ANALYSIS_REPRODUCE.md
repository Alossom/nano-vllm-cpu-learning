# Analysis reproduction

Run after the four measured stages and the separate profiling run.

```bash
.venv/bin/python - <<'PY'
import json,statistics,hashlib,subprocess
from pathlib import Path
p=Path("/mnt/c/Users/30262/Documents/Codex/2026-09-16/new-chat/outputs/benchmark-2026-11-01")
stages=["custom","hf_eager","hf_sdpa","custom_confirm"]
docs={k:json.loads((p/(k+".json")).read_text()) for k in stages}
summary={}
def subset(stage,case):
    return [r for r in docs[stage]["records"] if r["case"]==case]
def vals(rows,path):
    ret=[]
    for row in rows:
        for key in path.split("."): row=row[key]
        ret.append(row)
    return ret
def stats(v):
    return dict(median=statistics.median(v),min=min(v),max=max(v),n=len(v))
for stage,doc in docs.items():
    for case in dict.fromkeys(r["case"] for r in doc["records"]):
        rows=subset(stage,case)
        metrics=["elapsed_s","output_tps","ttft_s","peak_rss_mib"]
        if "dynamic" in case:
            metrics+=["requests.A.max_itl_s","requests.A.latency_s","requests.B.ttft_s"]
        elif "requests" in rows[0] and "A" in rows[0]["requests"]:
            metrics+=["requests.A.mean_itl_s"]
        summary[stage+"/"+case]={metric:stats(vals(rows,metric)) for metric in metrics}
checks={}
def tokens(row):
    if "requests" in row: return [v["tokens"] for k,v in sorted(row["requests"].items())]
    if "parts" in row: return [t for part in row["parts"] for t in tokens(part)]
    return row["tokens"]
def first(s,c):return subset(s,c)[0]
for stage,doc in docs.items():
    for case in {r["case"] for r in doc["records"]}:
        rows=subset(stage,case)
        checks[stage+"/"+case+"/repeat_outputs"]=all(tokens(r)==tokens(rows[0]) for r in rows)
for hf in ["hf_eager","hf_sdpa"]:
    for case in ["single_32","single_257_full"]:
        checks[hf+"/"+case+"/custom_match"]=tokens(first(hf,case))==tokens(first("custom",case))
    checks[hf+"/batch/custom_full_match"]=tokens(first(hf,"batch4_64"))==tokens(first("custom_confirm","batch4_64_full"))
for case in ["single_257_chunk128","prefix_257_hit256"]:
    checks[case+"/full_match"]=tokens(first("custom",case))==tokens(first("custom","single_257_full"))
checks["batch_vs_sequential"]=tokens(first("custom","batch4_64"))==tokens(first("custom","sequential4_64"))
checks["batch_full_vs_chunked"]=tokens(first("custom","batch4_64"))==tokens(first("custom_confirm","batch4_64_full"))
reference=tokens(first("custom","dynamic_original"))
for stage in ["custom","custom_confirm"]:
    for row in docs[stage]["records"]:
        if "dynamic" in row["case"]:
            checks[stage+"/"+row["case"]+"/"+str(row["repetition"])+"/original_match"]=tokens(row)==reference
assert all(checks.values()),checks
profile=json.loads((p/"profile.json").read_text())
prof_summary={}
for phase,d in profile.items():
    names=["aten::mm","aten::bmm","aten::copy_","aten::isfinite","KVCache.read","KVCache.write","LMHead"]
    prof_summary[phase]={r["name"]:dict(self_percent=100*r["self_cpu_us"]/d["sum_self_cpu_us"],
        total_percent=100*r["total_cpu_us"]/d["sum_self_cpu_us"]) for r in d["operators"] if r["name"] in names}
result=dict(summary=summary,checks=checks,all_checks_passed=all(checks.values()),profile=prof_summary,
    measured_runs=sum(len(x["records"]) for x in docs.values()))
(p/"summary.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
root="/home/x1o/ai-infra/projects/nano-vllm"
ev={
"lscpu":subprocess.check_output(["lscpu"],text=True),
"memory":subprocess.check_output(["free","-h"],text=True),
"git_status_after":subprocess.check_output(["git","status","--short"],cwd=root,text=True),
"original_source_unchanged":not subprocess.check_output(["git","diff","HEAD^","HEAD","--","nanovllm/engine/scheduler.py","nanovllm/engine/sequence.py","nanovllm/engine/block_manager.py"],cwd=root,text=True),
"model_config_sha256":hashlib.sha256(Path("/home/x1o/ai-infra/models/Qwen3-0.6B/config.json").read_bytes()).hexdigest()
}
(p/"environment.json").write_text(json.dumps(ev,indent=2))
print("git status after:",ev["git_status_after"])

PY
```

## Separate profiling run (not used for benchmark latencies)

```bash
.venv/bin/python - <<'PY'
import sys,json,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,"/home/x1o/ai-infra/projects/nano-vllm")
import torch
from nanovllm.models.model_config_learning import Qwen3ModelConfig
from nanovllm.models.model_learning import TinyCausalLM
from nanovllm.models.checkpoint_learning import load_qwen3_directory
from nanovllm.engine.engine_learning import Engine
from nanovllm.engine.model_runner_learning import GreedyModelRunner
from nanovllm.sampling_params import SamplingParams
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
p=Path("/home/x1o/ai-infra/models/Qwen3-0.6B")
out=Path("/mnt/c/Users/30262/Documents/Codex/2026-09-16/new-chat/outputs/benchmark-2026-11-01")
cfg=Qwen3ModelConfig.from_json(p/"config.json")
model=TinyCausalLM(**cfg.model_kwargs(8)).eval()
load_qwen3_directory(model,p)
config=SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=1024,eos=151645,kvcache_block_size=256,num_kvcache_blocks=8)
g=torch.Generator().manual_seed(2026)
ids=torch.randint(100,10000,(257,),generator=g).tolist()
def scoped(fn,name):
    def wrapped(*a,**kw):
        with torch.profiler.record_function(name):
            return fn(*a,**kw)
    return wrapped
for cache in model.caches:
    cache.read=scoped(cache.read,"KVCache.read")
    cache.write=scoped(cache.write,"KVCache.write")
model.compute_logits=scoped(model.compute_logits,"LMHead")
results={}
with torch.inference_mode():
    warm=Engine(config,GreedyModelRunner(model))
    warm.add_request(ids,SamplingParams(max_tokens=3,ignore_eos=True))
    while not warm.is_finished(): warm.step()
    for phase in ["prefill","decode"]:
        e=Engine(config,GreedyModelRunner(model))
        e.add_request(ids,SamplingParams(max_tokens=3,ignore_eos=True))
        if phase=="decode": e.step()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            e.step()
        rows=[]
        for v in prof.key_averages():
            rows.append(dict(name=v.key,self_cpu_us=v.self_cpu_time_total,
                total_cpu_us=v.cpu_time_total,calls=v.count))
        rows.sort(key=lambda r:r["self_cpu_us"],reverse=True)
        results[phase]=dict(sum_self_cpu_us=sum(r["self_cpu_us"] for r in rows),operators=rows)
        print(phase)
        print(prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=12))
(out/"profile.json").write_text(json.dumps(results,indent=2))

PY
```
