import argparse
import json
from itertools import islice

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from sae_gpt2.eval import load_sae, make_batches


@torch.no_grad()
def collect_top_examples(model, sae, scale, layer, n_batches=500, batch_size=32,
                         ctx_len=128, k=10, pre=20, post=3):
    """Per feature: top-k activations over held-out text, each with a token
    window (pre tokens before, the active token, post tokens after)."""
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_pre"
    eos = model.tokenizer.eos_token_id
    W = pre + 1 + post
    offs = torch.arange(W, device=device)

    top_vals = torch.full((sae.d_sae, k), float("-inf"), device=device)
    top_win = torch.zeros(sae.d_sae, k, W, dtype=torch.long, device=device)
    fires = torch.zeros(sae.d_sae, device=device)
    n_tok = 0

    for tokens in islice(make_batches(model.tokenizer, ctx_len, batch_size, device),
                         n_batches):
        _, cache = model.run_with_cache(tokens, names_filter=hook_name,
                                        stop_at_layer=layer + 1)
        a = cache[hook_name][:, 1:, :] * scale        # drop BOS, same scaling as training
        B, T, _ = a.shape
        f = sae.encode(a.reshape(-1, sae.d_model))    # [B*T, d_sae]
        fires += (f > 0).sum(0)
        n_tok += f.shape[0]

        vals, idx = f.topk(k, dim=0)                  # [k, d_sae]
        vals, idx = vals.T, idx.T                     # [d_sae, k]
        b, p = idx // T, idx % T + 1                  # context index, position in context
        padded = F.pad(tokens, (pre, post), value=eos)
        win = padded[b.unsqueeze(-1), p.unsqueeze(-1) + offs]   # [d_sae, k, W]

        all_vals = torch.cat([top_vals, vals], 1)
        all_win = torch.cat([top_win, win], 1)
        top_vals, sel = all_vals.topk(k, dim=1)
        top_win = all_win.gather(1, sel.unsqueeze(-1).expand(-1, -1, W))
    return top_vals, top_win, fires / n_tok, n_tok


@torch.no_grad()
def logit_effects(model, sae, k=10, chunk=1024):
    """Tokens whose logits a feature's decoder direction raises / lowers via W_U.
    This is the *direct* effect only; at layer 6 later blocks also matter."""
    up, down = [], []
    for i in range(0, sae.d_sae, chunk):
        lg = sae.W_dec[i:i + chunk] @ model.W_U
        lg = lg - lg.mean(-1, keepdim=True)           # softmax ignores a constant shift
        up.append(lg.topk(k, -1).indices)
        down.append((-lg).topk(k, -1).indices)
    return torch.cat(up), torch.cat(down)


def build_records(model, sae, vals, wins, density, up, down, n_examples, pre):
    cache = {}

    def s(i):
        if i not in cache:
            cache[i] = model.tokenizer.decode([i]).replace("\n", "↵")
        return cache[i]

    vals, density = vals.cpu(), density.cpu()
    wins, up, down = wins.cpu().tolist(), up.cpu().tolist(), down.cpu().tolist()
    recs = []
    for i in range(sae.d_sae):
        if density[i].item() == 0:
            continue
        ex = []
        for j in range(n_examples):
            v = vals[i, j].item()
            if v <= 0:
                break
            w = wins[i][j]
            ex.append({"a": round(v, 2),
                       "l": "".join(s(t) for t in w[:pre]),
                       "c": s(w[pre]),
                       "r": "".join(s(t) for t in w[pre + 1:])})
        recs.append({"id": i, "density": density[i].item(), "max": vals[i, 0].item(),
                     "ex": ex, "up": [s(t) for t in up[i]], "down": [s(t) for t in down[i]]})
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="features.json")
    ap.add_argument("--n_batches", type=int, default=500)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, scale, cfg = load_sae(args.ckpt, device)
    model = HookedTransformer.from_pretrained(cfg["model_name"], device=device)
    model.eval()

    pre = 20
    vals, wins, density, n_tok = collect_top_examples(
        model, sae, scale, cfg["layer"], n_batches=args.n_batches,
        ctx_len=cfg["ctx_len"], pre=pre)
    up, down = logit_effects(model, sae)
    recs = build_records(model, sae, vals, wins, density, up, down, args.examples, pre)

    print(f"tokens scanned: {n_tok:,}")
    print(f"mean L0 on this sample (sum of densities): {density.sum().item():.1f}")
    print(f"features that fired at least once: {len(recs)} / {sae.d_sae}")
    meta = {"ckpt": args.ckpt.split("/")[-2], "layer": cfg["layer"], "d_sae": sae.d_sae,
            "tokens": n_tok, "l0": density.sum().item()}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "features": recs}, f)
    print("wrote", args.out)