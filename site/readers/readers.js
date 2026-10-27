/* Arrival Readers (/readers): who reads which line of the Arrival film.
   The lines come from /data/readers-lines.json and the claims from /api/readers (site/api/readers.mjs).
   Taking a line returns a token, kept in this browser, so only this browser (or the organiser) can change that claim.
   /readers?admin asks for the organiser's key, then shows Rename / Clear on every claimed line and names.json. */
(() => {
  'use strict';

  const API = '/api/readers';
  const LINES_URL = '/data/readers-lines.json';
  const POLL_MS = 8000;
  const KEY = { name: 'arrival-readers-name', tokens: 'arrival-readers-tokens', admin: 'arrival-readers-admin-key' };

  const $ = (s, root = document) => root.querySelector(s);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;   // names and lines only ever go in as text
    return e;
  }

  // ---------- storage that never throws (private windows, blocked site data): this tab's memory stands in ----------
  const mem = new Map();
  let storageBlocked = false;
  function load(k) {
    try { const v = localStorage.getItem(k); return v !== null ? v : mem.has(k) ? mem.get(k) : null; }
    catch { storageBlocked = true; return mem.has(k) ? mem.get(k) : null; }
  }
  function save(k, v) {
    if (v == null) mem.delete(k); else mem.set(k, v);
    try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { storageBlocked = true; }
  }
  try { localStorage.setItem('arrival-readers-probe', '1'); localStorage.removeItem('arrival-readers-probe'); } catch { storageBlocked = true; }

  // ---------- state ----------
  const state = {
    lines: [],            // from readers-lines.json, in script order
    ready: false,         // lines are on the page
    loaded: false,        // the claims have arrived at least once
    closed: false,        // the API says sign-ups are not open (no store in production)
    memory: false,        // the API keeps claims in memory (a local or preview copy)
    offline: false,       // the last poll failed; we keep showing what we had
    adminMode: new URLSearchParams(location.search).has('admin'),
    admin: false,         // the organiser's key has been checked
  };
  let server = {};                   // n -> {name, sent, at}: the API's last word
  const inflight = new Map();        // n -> the claim as it will be (null: gone) while a change to it saves
  const rows = {};                   // n -> {li, st, slot, sig, form}
  const armed = new Map();           // n -> timer: "Give it back" / "Clear" waiting for its second press
  let tokens = readTokens();         // n -> {token, at}: the claims this browser made
  let adminKey = load(KEY.admin) || '';
  let writes = 0;                    // bumped when a change comes back, so a poll sent before it is ignored

  function readTokens() {
    try { const t = JSON.parse(load(KEY.tokens) || '{}'); return t && typeof t === 'object' && !Array.isArray(t) ? t : {}; }
    catch { return {}; }
  }
  function saveTokens() { save(KEY.tokens, JSON.stringify(tokens)); }
  function dropToken(n) { if (tokens[n]) { delete tokens[n]; saveTokens(); } }

  const view = n => (inflight.has(n) ? inflight.get(n) : server[n] || null);
  // Mine: this browser holds the token for this very claim (same claim time), not an older one on the same line.
  const isMine = n => { const t = tokens[n], c = view(n); return !!(t && c && c.at && t.at === c.at); };
  const MAX_LINES = 3;   // per person; the API refuses a fourth under the same name
  const heldHere = () => Object.keys(tokens).filter(isMine).length;

  // ---------- the lines, grouped by chapter ----------
  function buildRows() {
    const box = $('#chapters');
    box.replaceChildren();
    let chapter = null, ol = null, i = 0;
    for (const l of state.lines) {
      if (l.chapter !== chapter) {
        chapter = l.chapter;
        const sec = el('section', 'chapter'), h = el('h2', null, chapter);
        h.id = 'chapter-' + (++i);
        sec.setAttribute('aria-labelledby', h.id);
        ol = el('ol', 'rows');
        sec.append(h, ol);
        box.append(sec);
      }
      const li = el('li', 'row');
      li.id = 'r' + l.n;
      li.tabIndex = -1;
      const no = el('div', 'no');
      no.append(el('span', 'sr-only', 'Line '), l.n);
      const body = el('div', 'body'), meta = el('div', 'meta');
      meta.append(el('span', null, `at ${l.at} · say it in ${l.say} s`));
      if (l.card === 'big') meta.append(el('span', 'big', 'on screen large'));
      body.append(el('p', 'strip', l.text), meta);
      const claim = el('div', 'claim'), st = el('div', 'state'), slot = el('div', 'formslot');
      claim.append(st, slot);
      li.append(no, body, claim);
      ol.append(li);
      rows[l.n] = { li, st, slot, sig: '', form: null };
    }
    box.removeAttribute('aria-busy');
    const h = location.hash.slice(1), r = /^r\d{2,3}$/.test(h) && rows[h.slice(1)];
    if (r) { r.li.classList.add('target'); requestAnimationFrame(() => r.li.scrollIntoView({ block: 'center' })); }
  }

  // ---------- rendering ----------
  function renderAll() {
    for (const n in rows) renderRow(n);
    renderProgress();
    renderAdmin();
    renderNames();
    renderNotes();
  }

  function renderRow(n) {
    const r = rows[n], c = view(n), taken = !!(c && c.name), busy = inflight.has(n), own = isMine(n);
    const full = !state.admin && heldHere() >= MAX_LINES;
    const sig = JSON.stringify([state.loaded, state.closed, state.admin, taken && c.name, taken && !!c.sent, own, busy, armed.has(n), full]);
    if (sig === r.sig) return;        // unchanged: leave it alone, so focus and a half-typed name survive the poll
    r.sig = sig;
    const focused = r.st.contains(document.activeElement) ? document.activeElement.dataset.act : null;
    r.li.classList.toggle('taken', taken);
    r.st.replaceChildren();
    if (!state.closed && !state.loaded) {
      const p = el('span', 'pending', '…');
      p.setAttribute('aria-hidden', 'true');
      r.st.append(p);
    } else if (!state.closed && taken) {
      const tag = el('span', 'tag', c.name);
      if (busy && !c.at) tag.classList.add('saving');
      r.st.append(tag);
      if (c.sent) r.st.append(el('span', 'seal', 'take in'));
      if (own || state.admin) {
        const acts = el('div', 'acts');
        if (state.admin) acts.append(button('Rename', 'rename', n, () => openForm(n, 'rename'), 'ghost'));
        acts.append(button(c.sent ? 'Not in yet' : own ? 'I sent my take' : 'Take is in', 'sent', n, () => setSent(n, !c.sent), 'ghost'));
        const drop = own ? 'Give it back' : 'Clear';
        acts.append(armed.has(n)
          ? button(`Sure? ${drop}`, 'drop', n, () => (own ? release(n) : clear(n)), 'ghost warn')
          : button(drop, 'drop', n, () => arm(n), 'ghost'));
        if (busy) for (const b of acts.querySelectorAll('button')) b.setAttribute('aria-disabled', 'true');
        r.st.append(acts);
      }
    } else if (!state.closed) {
      r.st.append(el('span', 'open', 'open'));
      if (full) r.st.append(el('span', 'pending', `you have ${MAX_LINES} lines`));
      else {
        const b = button('Take this line', 'take', n, () => openForm(n, 'claim'));
        if (busy) b.setAttribute('aria-disabled', 'true');
        r.st.append(b);
      }
    }
    if (focused) (r.st.querySelector(`[data-act="${focused}"]`) || r.st.querySelector('button') || r.li).focus({ preventScroll: true });
  }

  // Buttons stay focusable while their line saves (aria-disabled), so keyboard focus is never dropped mid-change.
  function button(label, act, n, fn, kind) {
    const b = el('button', 'btn' + (kind ? ' ' + kind : ''), label);
    b.type = 'button';
    b.dataset.act = act;
    b.addEventListener('click', () => { if (b.getAttribute('aria-disabled') !== 'true') fn(); });
    return b;
  }

  function renderProgress() {
    const p = $('#progress');
    if (state.closed) { p.textContent = 'Sign-ups are not open yet. The script is all here to read.'; return; }
    if (!state.loaded || !state.ready) {
      p.textContent = state.offline ? 'Could not reach the sign-up sheet. Trying again…' : 'Checking who is reading which line…';
      return;
    }
    let taken = 0, sent = 0;
    for (const l of state.lines) { const c = view(l.n); if (c && c.name) { taken++; if (c.sent) sent++; } }
    p.replaceChildren(el('b', null, String(taken)), ` of ${state.lines.length} lines taken · `, el('b', null, String(sent)), sent === 1 ? ' take in' : ' takes in');
  }

  function renderNotes() {
    note('memory', state.memory && !state.closed ? 'Test copy: claims made here stay on this server only. They are not the real sign-up sheet.' : null);
    note('offline', state.offline && state.loaded && !state.closed ? 'Lost touch with the sign-up sheet. You are seeing the last claims we had; this page keeps trying.' : null);
    note('storage', storageBlocked && !state.closed ? 'This browser is not saving data for this page, so you can change a line you take only until you close this tab.' : null);
  }
  function note(key, msg) {
    const box = $('#notes');
    let p = box.querySelector(`[data-note="${key}"]`);
    if (!msg) { if (p) p.remove(); return; }
    if (!p) { p = el('p', 'note'); p.dataset.note = key; box.append(p); }
    if (p.textContent !== msg) p.textContent = msg;
  }

  // ---------- the claim form (also the organiser's rename form) ----------
  function openForm(n, mode) {
    closeForms();
    disarm(n, true);
    const r = rows[n], c = view(n);
    const f = el('form', 'claimform'), id = 'name-' + n, errId = id + '-err';
    f.noValidate = true;
    const forOther = mode === 'rename' || state.admin;
    const lab = el('label', null, forOther ? 'Reader’s name, as it should appear in the film' : 'Your name, as it should appear in the film');
    lab.htmlFor = id;
    const inp = el('input');
    Object.assign(inp, { id, name: 'name', maxLength: 60, required: true, spellcheck: false, autocomplete: forOther ? 'off' : 'name' });
    inp.setAttribute('aria-describedby', errId);
    inp.value = mode === 'rename' ? (c && c.name) || '' : forOther ? '' : load(KEY.name) || '';
    const err = el('p', 'err');
    err.id = errId;
    err.hidden = true;
    const ok = el('button', 'btn', mode === 'rename' ? 'Save name' : 'Take line ' + n);
    ok.type = 'submit';
    const acts = el('div', 'acts');
    acts.append(ok, button('Cancel', 'cancel', n, () => closeForm(n, true), 'ghost'));
    f.append(lab, inp, acts, err);
    f.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); closeForm(n, true); } });
    f.addEventListener('submit', async e => {
      e.preventDefault();
      if (ok.disabled) return;
      const name = inp.value.trim().replace(/\s+/g, ' ');
      const problem = !name ? 'Type a name first.'
        : /[\u0000-\u001f\u007f-\u009f\u2028\u2029\u202A-\u202E\u2066-\u2069]/.test(name) ? 'Use only letters, spaces and ordinary punctuation in the name.'
        : name.length > 60 ? 'Keep the name to 60 characters.' : '';
      if (problem) { err.textContent = problem; err.hidden = false; inp.setAttribute('aria-invalid', 'true'); inp.focus(); return; }
      err.hidden = true;
      inp.removeAttribute('aria-invalid');
      ok.disabled = true;
      const done = mode === 'rename' ? await rename(n, name) : await take(n, name);
      if (done) { closeForm(n, false); focusRow(n); }
      else if (rows[n].form) ok.disabled = false;
    });
    r.slot.append(f);
    r.form = mode;
    inp.focus();
  }
  function closeForm(n, refocus) {
    const r = rows[n];
    if (!r || !r.form) return;
    r.slot.replaceChildren();
    r.form = null;
    if (refocus) focusRow(n);
  }
  function closeForms() { for (const n in rows) closeForm(n, false); }
  function focusRow(n) { const r = rows[n]; if (r) (r.st.querySelector('button') || r.li).focus({ preventScroll: true }); }

  // A poll can show that someone else got there first, or that a line being renamed was cleared.
  function settleForms() {
    for (const n in rows) {
      const r = rows[n], c = view(n);
      if (!r.form || inflight.has(n)) continue;
      if (r.form === 'claim' && c && c.name && !isMine(n)) { closeForm(n, true); toast(`Line ${n} was just taken by ${c.name}. Pick another line.`); }
      else if (r.form === 'rename' && !(c && c.name)) { closeForm(n, true); toast(`Line ${n} is open again.`); }
    }
  }

  // ---------- "Give it back" and "Clear" ask twice ----------
  function arm(n) {
    disarm(n, false);
    armed.set(n, setTimeout(() => disarm(n, true), 5000));
    renderRow(n);
    announce(isMine(n) ? `Press again to give line ${n} back.` : `Press again to clear line ${n}.`);
  }
  function disarm(n, render) {
    const t = armed.get(n);
    if (!t) return;
    clearTimeout(t);
    armed.delete(n);
    if (render) renderRow(n);
  }

  // ---------- changes: shown at once, put right by the server's answer ----------
  async function send(n, next, body, asOrganiser) {
    if (inflight.has(n)) return { status: -1, data: {} };
    inflight.set(n, next);
    disarm(n, false);
    renderAll();
    const headers = { 'Content-Type': 'application/json' };
    if (asOrganiser && adminKey) headers['x-admin-key'] = adminKey;
    let status = 0, data = {};
    try {
      const res = await fetch(API, { method: 'POST', headers, body: JSON.stringify(body), cache: 'no-store' });
      status = res.status;
      data = (await res.json().catch(() => null)) || {};
    } catch { /* offline or the server is unreachable: status stays 0 */ }
    inflight.delete(n);          // whatever happens next, the optimistic version goes: rollback is the server's state
    writes++;
    if (status === 200 && body.action === 'claim' && typeof data.token === 'string') { tokens[n] = { token: data.token, at: data.at }; saveTokens(); }
    if (data.claims && typeof data.claims === 'object') apply(data, false);
    else renderAll();
    return { status, data };
  }

  async function take(n, name) {
    const r = await send(n, { name, sent: false, at: null }, { action: 'claim', n, name }, false);
    if (r.status === 200 && r.data.token) {
      if (!state.admin) save(KEY.name, name);
      announce(`Line ${n} is yours.`);
      return true;
    }
    if (r.status === 409 && r.data.error === 'limit') {
      closeForm(n, true);
      toast(r.data.message || `That name already has ${MAX_LINES} lines, the most one person can take.`);
      return false;
    }
    if (r.status === 409 && r.data.error === 'taken') {
      closeForm(n, true);
      const who = r.data.claim && r.data.claim.name;
      toast(who ? `Line ${n} was just taken by ${who}. Pick another line.` : `Line ${n} was just taken. Pick another line.`);
      return false;
    }
    return failed(n, r);
  }

  async function setSent(n, sent) {
    const c = view(n);
    if (!c) return;
    const own = isMine(n), body = { action: sent ? 'sent' : 'unsent', n };
    if (own) body.token = tokens[n].token;
    const r = await send(n, { ...c, sent }, body, !own && state.admin);
    if (r.status !== 200) failed(n, r);
  }

  async function release(n) {
    const t = tokens[n];
    if (!t) return;
    const r = await send(n, null, { action: 'release', n, token: t.token }, false);
    if (r.status === 200) { dropToken(n); renderAll(); announce(`You gave line ${n} back.`); return; }
    failed(n, r);
  }

  async function clear(n) {
    const r = await send(n, null, { action: 'clear', n }, true);
    if (r.status === 200) { announce(`Line ${n} is open again.`); return; }
    failed(n, r);
  }

  async function rename(n, name) {
    const c = view(n);
    if (!c) return false;
    const r = await send(n, { ...c, name }, { action: 'rename', n, name }, true);
    return r.status === 200 ? true : failed(n, r);
  }

  function failed(n, r) {
    const d = r.data || {}, msg = typeof d.message === 'string' ? d.message : '';
    if (r.status === -1) return false;   // that line is already saving
    if (r.status === 0) toast('That did not save. Check your connection and try again.');
    else if (r.status === 503 && d.error === 'not_configured') close();
    else if (r.status === 403 && /^(bad_admin_key|admin_off|admin_only)$/.test(d.error)) lockAdmin(msg);
    else if (r.status === 403 || r.status === 404) { dropToken(n); renderAll(); toast(msg || `Line ${n} has changed. This is how it stands now.`); }
    else toast(msg || 'That did not save. Try again in a moment.');
    return false;
  }

  // ---------- polling: every few seconds while the page is in view, and at once when it comes back ----------
  let polling = false, pollTimer = 0, lastPoll = 0;
  async function poll() {
    if (polling) return;
    polling = true;
    clearTimeout(pollTimer);
    const seq = writes;
    try {
      const res = await fetch(API, { cache: 'no-store', headers: { Accept: 'application/json' } });
      const data = await res.json().catch(() => null);
      if (res.ok && data && data.claims && typeof data.claims === 'object') {
        state.offline = false;
        if (seq === writes) apply(data, true); else renderAll();
      } else if (res.status === 503 && data && data.error === 'not_configured') {
        state.offline = false;
        close();
      } else { state.offline = true; renderAll(); }
    } catch { state.offline = true; renderAll(); }
    polling = false;
    lastPoll = Date.now();
    schedule();
  }
  function schedule() { clearTimeout(pollTimer); if (document.visibilityState !== 'hidden') pollTimer = setTimeout(poll, POLL_MS); }
  function wake() { if (document.visibilityState !== 'hidden' && !polling && Date.now() - lastPoll > 1500) poll(); }
  document.addEventListener('visibilitychange', () => (document.visibilityState === 'hidden' ? clearTimeout(pollTimer) : wake()));
  window.addEventListener('focus', wake);
  window.addEventListener('online', wake);
  // a line taken in another tab of this browser is this tab's too
  window.addEventListener('storage', e => { if (e.key === KEY.tokens) { tokens = readTokens(); renderAll(); } });

  function apply(data, fromPoll) {
    const next = {};
    for (const [n, c] of Object.entries(data.claims)) {
      if (c && typeof c.name === 'string' && c.name) next[n] = { name: c.name, sent: c.sent === true, at: typeof c.at === 'string' ? c.at : null };
    }
    server = next;
    state.loaded = true;
    state.closed = false;
    state.memory = data.store === 'memory';
    // A newer claim on a line this browser held means ours was cleared: forget that token. (A line that merely looks
    // open is not enough; that can be a poll that crossed with our own claim.)
    let stale = false;
    for (const n of Object.keys(tokens)) {
      const c = server[n], t = tokens[n];
      if (!t || typeof t.token !== 'string' || (c && c.at && t.at && c.at > t.at)) { delete tokens[n]; stale = true; }
    }
    if (stale) saveTokens();
    renderAll();
    if (fromPoll) settleForms();
  }

  function close() {
    state.closed = true;
    closeForms();
    renderAll();
  }

  // ---------- organiser mode: /readers?admin ----------
  function renderAdmin() {
    const box = $('#admin');
    box.hidden = !state.adminMode;
    if (!state.adminMode) return;
    $('#adminform').hidden = state.admin;
    $('#adminon').hidden = !state.admin;
  }
  function adminError(msg) {
    const e = $('#adminerr');
    e.textContent = msg || '';
    e.hidden = !msg;
  }
  async function verifyAdmin(key) {
    let status = 0, data = {};
    try {
      const res = await fetch(API, {
        method: 'POST', cache: 'no-store',
        headers: { 'Content-Type': 'application/json', 'x-admin-key': key },
        body: JSON.stringify({ action: 'verify' }),
      });
      status = res.status;
      data = (await res.json().catch(() => null)) || {};
    } catch { /* status 0 */ }
    if (status === 200) {
      adminKey = key;
      save(KEY.admin, key);
      state.admin = true;
      adminError('');
      $('#adminkey').value = '';
      closeForms();
      renderAll();
      return true;
    }
    state.admin = false;
    if (status === 403) { adminKey = ''; save(KEY.admin, null); }
    else $('#adminkey').value = key;   // keep it typed in, so trying again is one press
    adminError(status === 0 ? 'Could not reach the server to check the key. Try again.' : data.message || 'Could not check the key. Try again.');
    renderAll();
    return false;
  }
  function lockAdmin(msg) {
    state.admin = false;
    adminKey = '';
    save(KEY.admin, null);
    closeForms();
    renderAll();
    if (state.adminMode) { adminError(msg || ''); $('#adminkey').focus(); }
    else if (msg) toast(msg);
  }
  $('#adminform').addEventListener('submit', async e => {
    e.preventDefault();
    const key = $('#adminkey').value.trim(), go = $('#adminform button');
    if (!key) { adminError('Type the admin key.'); $('#adminkey').focus(); return; }
    go.disabled = true;
    adminError('');
    await verifyAdmin(key);
    go.disabled = false;
    if (state.admin) $('#adminlock').focus();
  });
  $('#adminlock').addEventListener('click', () => { lockAdmin(''); announce('Organiser mode is locked.'); });

  // ---------- names.json for tools/assemble.py ----------
  function renderNames() {
    const box = $('#forasm');
    box.hidden = !state.admin;
    if (box.hidden) return;
    // written out by hand to keep "01" ... "37" in order (an object would put "10" and up before "01")
    const pairs = state.lines.map(l => [l.n, view(l.n)]).filter(([, c]) => c && c.name);
    const text = pairs.length ? '{\n' + pairs.map(([n, c]) => `  ${JSON.stringify(n)}: ${JSON.stringify(c.name)}`).join(',\n') + '\n}' : '{}';
    const pre = $('#names');
    if (pre.textContent !== text) pre.textContent = text;
  }
  $('#copy').addEventListener('click', () => {
    const pre = $('#names'), text = pre.textContent, fl = $('#flash');
    const fallback = () => {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const s = getSelection();
      s.removeAllRanges();
      s.addRange(range);
      fl.textContent = 'Selected: press ⌘C or Ctrl+C';
    };
    try { navigator.clipboard.writeText(text).then(() => { fl.textContent = 'Copied'; }, fallback); } catch { fallback(); }
  });

  // ---------- messages ----------
  let toastTimer = 0;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      t.classList.remove('show');
      setTimeout(() => { if (!t.classList.contains('show')) t.textContent = ''; }, 250);
    }, 4200);
  }
  function announce(msg) {
    const s = $('#sr');
    s.textContent = '';
    setTimeout(() => { s.textContent = msg; }, 30);
  }

  // ---------- start ----------
  async function loadLines() {
    for (let attempt = 0; ; attempt++) {
      try {
        const res = await fetch(LINES_URL, { cache: 'no-cache' });   // revalidate, so a new script shows on reload
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const lines = (Array.isArray(data.lines) ? data.lines : [])
          .filter(l => l && /^\d{2,3}$/.test(l.n) && typeof l.text === 'string')
          .sort((a, b) => Number(a.n) - Number(b.n));
        if (!lines.length) throw new Error('no lines');
        return { lines, length: typeof data.length === 'string' ? data.length : null };
      } catch (e) {
        if (attempt >= 2) throw e;
        await new Promise(r => setTimeout(r, 700 * (attempt + 1)));
      }
    }
  }

  renderAll();
  poll();
  if (state.adminMode && adminKey) verifyAdmin(adminKey);
  else if (state.adminMode) requestAnimationFrame(() => $('#adminkey').focus({ preventScroll: true }));

  loadLines().then(({ lines, length }) => {
    state.lines = lines;
    for (const s of document.querySelectorAll('[data-count]')) s.textContent = String(lines.length);
    if (length) for (const s of document.querySelectorAll('[data-length]')) s.textContent = length;
    buildRows();
    state.ready = true;
    renderAll();
  }, () => {
    const box = $('#chapters');
    box.removeAttribute('aria-busy');
    box.replaceChildren(el('p', 'note', 'The script did not load. Reload the page to try again.'));
  });
})();
