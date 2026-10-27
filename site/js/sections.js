// Fills every number on the page from the exported data. Nothing here is typed in by hand except wording.
import { FAMILIES, frameUrl } from './data.js';
import { fmt } from './util.js';

const LABEL = { open_drawer: 'Open drawer', open_door: 'Open cabinet door', open_fridge: 'Open fridge', put_in_drawer: 'Put in drawer', wipe_counter: 'Wipe counter', counter_to_sink: 'Counter to sink' };
export const familyLabel = (f) => LABEL[f] || f;

const FORMATS = {
  int: (v) => fmt.n(v),
  len: (v) => (Array.isArray(v) ? v.length : Object.keys(v || {}).length),
  keys: (v) => Object.keys(v || {}).length + ' kinds',
  hz: (v) => `${fmt.n(v)} Hz`,
  okword: (v) => (v ? 'ok' : 'not ok'),
  source: (v) => (v === 'authored_sample' ? 'authored sample' : v === 'roomplan' ? 'iPhone RoomPlan' : String(v)),
  bounds: (v) => `${(v.max[0] - v.min[0]).toFixed(1)} x ${(v.max[1] - v.min[1]).toFixed(1)} x ${(v.max[2] - v.min[2]).toFixed(1)} m`,
  dims: (v) => `${v.state_dim} / ${v.action_dim}`,
  cams: (v) => `${v.cameras.join(' + ')} at ${v.resolution.join('x')}`,
  res: (v) => v.join('×'),
};

function resolve(root, path) { return path.split('.').reduce((o, k) => (o == null ? o : o[k]), root); }

export function bindData(data) {
  const roots = { rooms: data.rooms, stats: data.stats, scan: { n: data.scan.n, ...data.scan.meta } };
  document.querySelectorAll('[data-bind]').forEach((el) => {
    const [path, f] = el.dataset.bind.split('|');
    let v = resolve(roots, path);
    if (v === undefined) return;
    if (f && f.startsWith('count:')) { const kind = f.slice(6); v = v.filter((p) => p.kind === kind).length; }
    else if (f && FORMATS[f]) v = FORMATS[f](v);
    el.textContent = String(v);
  });
}

export function renderEvaluate(stats) {
  const ev = stats.eval, rows = document.getElementById('evalRows'), total = document.getElementById('evalTotal');
  rows.innerHTML = '';
  const pct = (x) => (x * 100).toFixed(3) + '%';
  const ci = (lo, hi, pt) => `<div class="axis" role="img" aria-label="95% Wilson interval ${lo.toFixed(2)} to ${hi.toFixed(2)}, observed ${pt.toFixed(2)}"><span class="axis__ci" style="left:${pct(lo)};width:${pct(hi - lo)}"><b></b></span><span class="axis__pt" style="left:${pct(pt)}"></span></div>`;
  for (const f of FAMILIES) {
    const e = ev.families[f]; if (!e) continue;
    const arm = e.arm_contact_episodes ? ` · arm touched furniture ${(e.arm_contact_max_depth_m * 1000).toFixed(1)} mm` : '';
    const row = document.createElement('div');
    row.className = 'erow';
    row.innerHTML = `<div class="erow__name">${familyLabel(f)}<small>${stats.families[f].instruction}</small></div>${ci(e.wilson_95[0], e.wilson_95[1], e.success_rate)}
      <div class="erow__fig"><strong>${e.successes} / ${e.episodes}</strong><small>closest person ${e.min_human_distance_m.toFixed(2)} m · ${e.human_contacts} contacts${arm}</small></div>`;
    rows.appendChild(row);
  }
  const t = ev.totals, per = ev.episodes_per_family;
  // the lead is built from the data, so "every episode succeeded" and "one per family" can never drift from stats.json
  const lead = document.getElementById('evLead');
  if (lead) lead.innerHTML = `The evaluation harness ran <b>${per}</b> ${per === 1 ? 'episode' : 'episodes'} per task family with the <b>${ev.policy}</b> expert and <b>${ev.people}</b> people walking. ${t.successes === t.episodes ? 'Every episode succeeded.' : `${t.successes} of ${t.episodes} episodes succeeded.`} ${per === 1 ? 'One episode per family shows that the pipeline runs and says nothing about a success rate.' : `${per} episodes per family still give wide intervals.`}`;
  total.innerHTML = `<div class="erow__name">All families<small>pooled, still only ${t.episodes} runs</small></div>${ci(t.wilson_95[0], t.wilson_95[1], t.success_rate)}
    <div class="erow__fig"><strong>${t.successes} / ${t.episodes}</strong><small>${t.human_contacts} human contacts · ${t.arm_contact_episodes} runs touched furniture</small></div>`;

  const minD = Math.min(...Object.values(ev.families).map((e) => e.min_human_distance_m));
  const maxArm = Math.max(...Object.values(ev.families).map((e) => e.arm_contact_max_depth_m));
  const f1 = ev.families[FAMILIES[0]];
  document.getElementById('caveats').innerHTML = [
    `<b>${per === 1 ? 'One run per task.' : `${per} runs per task.`}</b> ${f1.successes} of ${f1.episodes} succeeding has a 95% Wilson interval from ${f1.wilson_95[0].toFixed(2)} to ${f1.wilson_95[1].toFixed(2)}. Pooled, ${t.successes} of ${t.episodes} runs have an interval from ${t.wilson_95[0].toFixed(2)} to ${t.wilson_95[1].toFixed(2)}.`,
    `<b>The policy is a scripted expert.</b> The harness ran <code>policy: ${ev.policy}</code>, so no learned model was involved. Nothing here measures GR00T.`,
    `<b>Not held out.</b> <code>holdout: ${ev.holdout}</code>, with ${ev.people} people walking in the room.`,
    `<b>One authored room.</b> The twin is the sample kitchen. Safety figures apply to this room only.`,
    `<b>Contacts.</b> ${t.human_contacts} contacts with people; the closest approach was ${minD.toFixed(2)} m. ${t.arm_contact_episodes} of ${t.episodes} runs brushed furniture, at most ${(maxArm * 1000).toFixed(1)} mm deep.`,
  ].map((s) => `<li>${s}</li>`).join('');

  const c = stats.cost, rep = stats.dataset.report;
  const costLine = (cmp) => {
    if (!cmp.measured) return `<b>Cost against teleoperation.</b> Not compared: ${cmp.reason}, so no saving is claimed.`;
    const g = cmp.generated, tl = cmp.teleop;
    const time = g.seconds_per_success == null ? 'generated time per success was not recorded'
      : `${g.seconds_per_success} s of local generation per success (n=${g.n_success}) against ${tl.seconds_per_success} s of operator time (n=${tl.n_success}), so teleop took ${cmp.seconds_ratio_teleop_over_generated}x as long`;
    return `<b>Cost against teleoperation.</b> On <code>${cmp.task}</code> in ${cmp.room}: ${time}. Machine seconds against human seconds, no dollar saving is claimed.`;
  };
  document.getElementById('todo').innerHTML = [
    `<b>GR00T policy server and the held-out comparison (G2, G3).</b> The fine-tune ran (G1, self-reported, small samples). The policy shim has not run against a real GR00T server, and the protocol that compares the base model, generic data and this-room data at 50 episodes per cell has produced no results yet.`,
    `<b>Measured real kitchens (P1, P2).</b> One kitchen scanned with the capture app on an iPhone 13 Pro Max has run end to end, but it has no tape measurements, the RGB-D recorder is not done, and the point cloud on this page is simulated.`,
    costLine(c.comparison),
  ].map((s) => `<li>${s}</li>`).join('');

  document.getElementById('measured').innerHTML = `
    <div><p class="mono">Local CPU cost</p><p>$${c.dollars.local_cpu.usd}</p><p>Measured over ${rep.attempts} attempts on the local CPU. Other cost lines are not measured.</p></div>
    <div><p class="mono">Throughput</p><p>${fmt.n(rep.demos_per_hour, 1)} <small class="mono">demos/h</small></p><p>${rep.successes} demos in ${rep.seconds} s on the local CPU.</p></div>
    <div><p class="mono">Dropped</p><p>${rep.dropped} of ${rep.attempts}</p><p>${rep.dropped === 0 && rep.successes === rep.attempts ? 'Every attempt passed the physics success check.' : `${rep.successes} of ${rep.attempts} attempts passed the physics success check.`}</p></div>`;

  // print the loader's own warnings, code and message, rather than an explanation written for one particular code
  const warns = stats.dataset.loader_check.warnings;
  const warnEl = document.getElementById('warnText');
  if (warnEl) warnEl.textContent = warns.map((w) => `${w.code}: ${w.msg}`).join(' ') || 'No warnings.';
}

export function renderFallback(stats) {
  const grid = document.getElementById('fallbackGrid'); grid.innerHTML = '';
  for (const f of FAMILIES) {
    const fig = document.createElement('figure');
    fig.innerHTML = `<div>${['ego_view_mid', 'ego_view_end', 'wrist_mid', 'wrist_end'].map((k) => { const i = k.lastIndexOf('_'); return `<img loading="lazy" width="128" height="128" alt="${k.replace(/_/g, ' ')} frame of ${familyLabel(f)}" src="${frameUrl(f, k.slice(0, i), k.slice(i + 1))}">`; }).join('')}</div>
      <figcaption>${stats.families[f].instruction}<small>${familyLabel(f)} · ${stats.families[f].duration_s} s · ${stats.families[f].success ? 'success' : 'failed'}</small></figcaption>`;
    grid.appendChild(fig);
  }
}
