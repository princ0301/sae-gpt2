import argparse
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT-2 small SAE features</title>
<style>
:root{color-scheme:light dark;--mut:#888;--hl:#ffd54a66}
body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px}
.card{border:1px solid #8884;border-radius:8px;padding:10px 14px;margin:10px 0}
.meta{color:var(--mut);font-size:13px}
.ex{font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap;margin:3px 0}
mark{background:var(--hl);color:inherit;border-radius:3px}
input,select,button{font:inherit;padding:4px 8px}
</style></head><body>
<h1>GPT-2 small: SAE features</h1>
<p class="meta" id="meta"></p>
<p class="meta">Activations are in the SAE's scaled units. "Promotes / suppresses" is the
direct logit effect of the decoder direction only, so treat it as a hint for a
layer-6 feature, not a verdict. Labels are hypotheses supported by the examples shown.</p>
<p><input id="q" placeholder="feature id or token text">
<select id="sort">
<option value="density">most frequent first</option>
<option value="rare">rarest first</option>
<option value="max">highest max activation</option>
<option value="id">feature id</option>
</select></p>
<div class="meta" id="info"></div>
<div id="list"></div>
<p><button id="more">Show more</button></p>
<script>
const DATA = __DATA__;
const F = DATA.features;
const $ = id => document.getElementById(id);
let shown = 50, view = F;
$('meta').textContent = 'Checkpoint ' + DATA.meta.ckpt + ', layer ' + DATA.meta.layer +
  ', ' + DATA.meta.d_sae + ' features, stats from ' + DATA.meta.tokens.toLocaleString() +
  ' held-out tokens (mean L0 ' + DATA.meta.l0.toFixed(1) + ').';
function card(f){
  const d = document.createElement('div'); d.className = 'card';
  const t = document.createElement('b'); t.textContent = 'Feature ' + f.id;
  const m = document.createElement('span'); m.className = 'meta';
  m.textContent = '  density ' + (f.density * 100).toFixed(3) + '%, max act ' + f.max.toFixed(1);
  const lg = document.createElement('div'); lg.className = 'meta';
  lg.textContent = 'promotes: ' + f.up.join(' ') + '  |  suppresses: ' + f.down.join(' ');
  d.append(t, m, lg);
  f.ex.forEach(e => {
    const p = document.createElement('div'); p.className = 'ex';
    const mk = document.createElement('mark'); mk.textContent = e.c;
    p.append(e.a.toFixed(1) + '  ', e.l, mk, e.r);
    d.append(p);
  });
  return d;
}
function render(){
  $('list').replaceChildren(...view.slice(0, shown).map(card));
  $('info').textContent = view.length + ' features match';
  $('more').style.display = shown < view.length ? '' : 'none';
}
function apply(){
  const q = $('q').value.trim().toLowerCase(), s = $('sort').value;
  view = F.filter(f => !q || String(f.id) === q ||
    f.up.join(' ').toLowerCase().includes(q) ||
    f.ex.some(e => e.c.toLowerCase().includes(q)));
  const key = {density:(a,b)=>b.density-a.density, rare:(a,b)=>a.density-b.density,
               max:(a,b)=>b.max-a.max, id:(a,b)=>a.id-b.id}[s];
  view = view.slice().sort(key);
  shown = 50; render();
}
$('q').addEventListener('input', apply);
$('sort').addEventListener('change', apply);
$('more').addEventListener('click', () => { shown += 50; render(); });
apply();
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="dashboard/index.html")
    args = ap.parse_args()

    data = json.load(open(args.data))
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")