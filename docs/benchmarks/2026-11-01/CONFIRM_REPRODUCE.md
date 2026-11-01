# Confirmation run

This run matches HF's whole-batch prefill and alternates original/new scheduling order. Whole-batch case: 1 warmup + 3 measured runs. Dynamic cases: 1 warmup each + 6 measured runs each. Both schedulers use the same CPU GreedyModelRunner and weights; the original GPU executor is not used.

```bash
.venv/bin/python -u - custom_confirm <<'PY'
import os, sys, json, time, statistics, resource, platform, subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
import transformers
ROOT = Path("/home/x1o/ai-infra/projects/nano-vllm")
sys.path.insert(0, str(ROOT))
MODEL = Path("/home/x1o/ai-infra/models/Qwen3-0.6B")
OUT = Path("/mnt/c/Users/30262/Documents/Codex/2026-09-16/new-chat/outputs/benchmark-2026-11-01")
stage = sys.argv[1]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(2026)
torch.set_grad_enabled(False)
metadata = dict(stage=stage, python=sys.version, torch=torch.__version__,
    transformers=transformers.__version__, platform=platform.platform(),
    threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
    cuda=torch.cuda.is_available(), commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    upstream_base=subprocess.check_output(["git","rev-parse","HEAD^"],cwd=ROOT,text=True).strip(),
    git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
    dtype="float32", repetitions=3, warmups_per_case=1, output_tokens=16,
    prompt_kind="deterministic synthetic in-vocabulary IDs, not quality evaluation",
    timing_exclusions="model loading, tokenization, warmup, prefix population; engine creation except sequential inner engines",
    model_config=json.loads((MODEL/"config.json").read_text()))
# Different seeds avoid accidental common prefixes between independent requests.
def prompt(n, variant=0):
    g = torch.Generator().manual_seed(2026+variant)
    return torch.randint(100, 10000, (n,), generator=g).tolist()
records = []
def emit(name, fn):
    for rep in range(-1, 3):
        result = fn()
        result.update(case=name, repetition=rep)
        if rep >= 0:
            records.append(result)
            (OUT/(stage+".json")).write_text(json.dumps(dict(metadata=metadata, records=records),ensure_ascii=False,indent=2))
        print(json.dumps(dict(case=name,rep=rep,elapsed_s=result["elapsed_s"],
            output_tps=result.get("output_tps"),ttft_s=result.get("ttft_s"),
            peak_rss_mib=result["peak_rss_mib"])),flush=True)
def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if stage.startswith("custom"):
    from nanovllm.models.model_config_learning import Qwen3ModelConfig
    from nanovllm.models.model_learning import TinyCausalLM
    from nanovllm.models.checkpoint_learning import load_qwen3_directory
    from nanovllm.engine.engine_learning import Engine, StepOutput
    from nanovllm.engine.model_runner_learning import GreedyModelRunner
    from nanovllm.engine.scheduler import Scheduler as OriginalScheduler
    from nanovllm.engine.sequence import Sequence as OriginalSequence
    from nanovllm.sampling_params import SamplingParams
    cfg = Qwen3ModelConfig.from_json(MODEL/"config.json")
    t = time.perf_counter()
    model = TinyCausalLM(**cfg.model_kwargs(num_blocks=8,block_size=256)).eval()
    report = load_qwen3_directory(model, MODEL)
    metadata.update(load_seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in model.parameters()),
        kv_cache_bytes=cfg.kv_cache_bytes(8),load_report=str(report))
    print("Loaded custom", metadata["load_seconds"],flush=True)
    class OriginalEngine(Engine):
        def __init__(self, config, runner):
            self.scheduler=OriginalScheduler(config)
            self.runner=runner
            self._completed={}
        def add_request(self, token_ids, sampling_params=None):
            seq=OriginalSequence(token_ids,sampling_params)
            self.scheduler.add(seq)
            return seq.seq_id
    def make_engine(policy="alternating",budget=128):
        config=SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=budget,eos=151645,
            kvcache_block_size=256,num_kvcache_blocks=8)
        runner=GreedyModelRunner(model)
        if policy=="original":
            e=OriginalEngine(config,runner)
        else:
            e=Engine(config,runner,policy=policy)
        e.measure={"prefill_tokens":0,"decode_tokens":0,"runner_seconds":0.0,"runner_prefill_seconds":0.0,"runner_decode_seconds":0.0}
        old=runner.run
        def run(seqs,is_prefill):
            count=sum(s.num_scheduled_tokens for s in seqs)
            start=time.perf_counter()
            result=old(seqs,is_prefill)
            elapsed=time.perf_counter()-start
            kind="prefill" if is_prefill else "decode"
            e.measure[kind+"_tokens"]+=count
            e.measure["runner_seconds"]+=elapsed
            e.measure["runner_"+kind+"_seconds"]+=elapsed
            return result
        runner.run=run
        return e
    def add(e,ids,n):
        return e.add_request(ids, SamplingParams(temperature=1.0,max_tokens=n,ignore_eos=True))
    def drive(e, entries, arrival=None):
        for key in e.measure: e.measure[key]=0
        registered={}
        times={}
        sequences={}
        traces=[]
        start=time.perf_counter()
        def submit(name,ids,n):
            rid=add(e,ids,n)
            registered[rid]=(name,time.perf_counter())
            times[name]=[]
        for name,ids,n in entries: submit(name,ids,n)
        step=0
        while not e.is_finished():
            if arrival is not None and step==2:
                submit(*arrival)
            t0=time.perf_counter()
            events=e.step()
            now=time.perf_counter()
            trace={"step":step,"start_s":t0-start,"end_s":now-start,"events":[]}
            for ev in events:
                name,admit=registered[ev.seq_id]
                trace["events"].append(dict(name=name,prefill=ev.is_prefill,scheduled=ev.scheduled_tokens,token=ev.token_id,finished=ev.finished))
                if ev.token_id is not None: times[name].append(now)
                if ev.finished: sequences[name]=list(e.take_result(ev.seq_id))
            traces.append(trace)
            step+=1
        elapsed=time.perf_counter()-start
        requests={}
        for rid,(name,admit) in registered.items():
            ts=times[name]
            gaps=[b-a for a,b in zip(ts,ts[1:])]
            requests[name]=dict(ttft_s=ts[0]-admit,latency_s=ts[-1]-admit,
                mean_itl_s=statistics.mean(gaps) if gaps else None,
                max_itl_s=max(gaps) if gaps else None,tokens=sequences[name])
        bm=e.scheduler.block_manager
        if hasattr(bm,"pool"): bm.pool.check_invariants()
        assert len(bm.free_block_ids)==8, "blocks not returned"
        n=sum(len(v) for v in sequences.values())
        return dict(elapsed_s=elapsed,output_tps=n/elapsed,
            ttft_s=statistics.mean(v["ttft_s"] for v in requests.values()),
            requests=requests,traces=traces,measure=dict(e.measure),peak_rss_mib=peak())
    def single(length,budget=128,policy="alternating"):
        return drive(make_engine(policy,budget),[("A",prompt(length),16)])
    def prefix():
        e=make_engine()
        seed=prompt(257)
        add(e,seed[:-1]+[10001],1)
        while not e.is_finished(): e.step()
        return drive(e,[("A",seed,16)])
    def batched():
        return drive(make_engine(),[(str(i),prompt(64,i),16) for i in range(4)])
    def sequential():
        parts=[]
        t=time.perf_counter()
        for i in range(4):
            parts.append(drive(make_engine(),[(str(i),prompt(64,i),16)]))
        elapsed=time.perf_counter()-t
        return dict(elapsed_s=elapsed,output_tps=64/elapsed,ttft_s=statistics.mean(p["ttft_s"] for p in parts),
            parts=parts,peak_rss_mib=peak())
    def dynamic(policy):
        return drive(make_engine(policy),[("A",prompt(32),16)],arrival=("B",prompt(513,4),8))
    metadata["repetitions"]=6
    metadata["confirmation"]="alternating execution order; three pairs original-first, three pairs current-first"
    with torch.inference_mode():
        # Match HF: all four 64-token requests share one prefill execution.
        emit("batch4_64_full",lambda:drive(make_engine(budget=1024),[(str(i),prompt(64,i),16) for i in range(4)]))
        for policy in ["original","alternating"]:
            dynamic(policy)
        for rep in range(6):
            order=["original","alternating"] if rep%2==0 else ["alternating","original"]
            for policy in order:
                result=dynamic(policy)
                result.update(case="paired_dynamic_"+policy,repetition=rep)
                records.append(result)
                (OUT/(stage+".json")).write_text(json.dumps(dict(metadata=metadata,records=records),ensure_ascii=False,indent=2))
                print(json.dumps(dict(case=result["case"],rep=rep,elapsed_s=result["elapsed_s"],
                    A_max_itl=result["requests"]["A"]["max_itl_s"],
                    B_ttft=result["requests"]["B"]["ttft_s"])),flush=True)

else:
    from transformers import AutoModelForCausalLM
    implementation=stage.removeprefix("hf_")
    t=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,
        attn_implementation=implementation).eval()
    metadata.update(load_seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in model.parameters()),
        attention_implementation=model.config._attn_implementation)
    print("Loaded HF",implementation,metadata["load_seconds"],flush=True)
    def hf_run(length,batch=1):
        ids=torch.tensor([prompt(length,i) for i in range(batch)],dtype=torch.long)
        cache=None
        sequences=[[] for _ in range(batch)]
        times=[]
        t=time.perf_counter()
        for step in range(16):
            out=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values
            ids=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
            tokens=ids[:,0].tolist()
            times.append(time.perf_counter())
            for seq,tok in zip(sequences,tokens): seq.append(tok)
        elapsed=time.perf_counter()-t
        gaps=[b-a for a,b in zip(times,times[1:])]
        return dict(elapsed_s=elapsed,output_tps=16*batch/elapsed,ttft_s=times[0]-t,
            mean_itl_s=statistics.mean(gaps),max_itl_s=max(gaps),tokens=sequences,peak_rss_mib=peak())
    with torch.inference_mode():
        for name,length,batch in [("single_32",32,1),("single_257_full",257,1),("batch4_64",64,4)]:
            emit(name,lambda length=length,batch=batch:hf_run(length,batch))
print("DONE",stage,flush=True)

PY
```
