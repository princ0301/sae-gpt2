import argparse
import time
from pathlib import Path

import torch
import yaml
from transformer_lens import HookedTransformer

from sae_gpt2.data import ActivationBuffer, token_stream
from sae_gpt2.model import SAE


def lr_mult(step: int, total: int, warmup: int, decay: int) -> float:
    """Linear warmup, constant, then linear decay to 0 over the last `decay` steps."""
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - decay:
        return max(0.0, (total - step) / max(decay, 1))
    return 1.0


def l1_mult(step: int, warmup: int) -> float:
    """Linear warmup of the sparsity coefficient from 0 to 1."""
    return min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0


def save_ckpt(path, sae, cfg, scale, step, tokens_seen):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "cfg": cfg,
        "scale": scale,  # MUST be reused at eval/dashboard time
        "step": step,
        "tokens_seen": tokens_seen,
        "d_model": sae.d_model,
        "d_sae": sae.d_sae,
    }, path)


def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model = HookedTransformer.from_pretrained(cfg["model_name"], device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stream = token_stream(model.tokenizer, ctx_len=cfg["ctx_len"], heldout=False)
    buf = ActivationBuffer(
        model, stream, layer=cfg["layer"], buffer_tokens=cfg["buffer_tokens"],
        batch_size=cfg["batch_size"], gen_batch=cfg["gen_batch"],
        device=device, seed=cfg["seed"],
    )
    scale = buf.estimate_scale()
    print(f"activation scale = {scale:.4f}")

    sae = SAE(model.cfg.d_model, cfg["d_sae"], seed=cfg["seed"]).to(device)
    sae.init_b_dec(next(buf).mean(0))  # data mean from one normalized batch

    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])

    bs = cfg["batch_size"]
    total_steps = cfg["total_tokens"] // bs
    lr_warm = int(cfg["lr_warmup_frac"] * total_steps)
    lr_decay = int(cfg["lr_decay_frac"] * total_steps)
    l1_warm = int(cfg["l1_warmup_frac"] * total_steps)
    dead_steps = max(1, cfg["dead_window_tokens"] // bs)
    since_fired = torch.zeros(cfg["d_sae"], device=device)

    run = None
    if cfg["wandb"]:
        import wandb
        run = wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg)

    ckpt_dir = Path(cfg["ckpt_dir"]) / cfg["run_name"]
    t0 = time.time()
    step = -1
    for step, x in zip(range(total_steps), buf):
        lr = cfg["lr"] * lr_mult(step, total_steps, lr_warm, lr_decay)
        for g in opt.param_groups:
            g["lr"] = lr
        l1_coeff = cfg["l1_coeff"] * l1_mult(step, l1_warm)

        loss, m = sae.loss(x, l1_coeff)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()  # project out grad along decoder rows
        opt.step()
        sae.normalize_decoder()     # re-enforce unit norm

        since_fired += 1
        since_fired[sae.last_active] = 0
        tokens_seen = (step + 1) * bs

        if (step + 1) % cfg["log_every"] == 0:
            dead_frac = (since_fired > dead_steps).float().mean().item()
            tps = tokens_seen / (time.time() - t0)
            log = {**m, "lr": lr, "l1_coeff": l1_coeff, "dead_frac": dead_frac,
                   "tokens_per_sec": tps, "tokens": tokens_seen}
            print(f"step {step+1}/{total_steps} | mse {m['mse']:.2f} | l1 {m['l1']:.2f} | "
                  f"L0 {m['l0']:.1f} | EV {m['explained_var']:.3f} | "
                  f"dead {dead_frac:.3f} | {tps:,.0f} tok/s")
            if run:
                run.log(log, step=step + 1)

        if (step + 1) % cfg["ckpt_every"] == 0:
            save_ckpt(ckpt_dir / f"step_{step+1}.pt", sae, cfg, scale, step + 1, tokens_seen)

    save_ckpt(ckpt_dir / "final.pt", sae, cfg, scale, step + 1, (step + 1) * bs)
    print(f"saved {ckpt_dir / 'final.pt'}")
    if run:
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[],
                    help="override config values, e.g. --set l1_coeff=2 d_sae=24576")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    train(cfg)