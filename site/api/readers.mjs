// The readers' sign-up sheet for the Arrival film. The page is site/readers.html, served at /readers.
//
//   GET  /api/readers   200 {claims: {"07": {name, sent, at}}}   (token hashes never leave the server)
//   POST /api/readers   JSON body, Content-Type: application/json
//     {action: "claim", n, name}                        200 {token, at, claims} | 409 taken | 409 limit (3 lines a name)
//     {action: "sent" | "unsent" | "release", n, token} 200 {claims} | 403 not this browser's line | 404 nobody has it
//     {action: "rename", n, name} | {action: "clear", n} | {action: "verify"}
//         need the header x-admin-key equal to READERS_ADMIN_KEY (if that is unset, they are refused). With the key,
//         sent / unsent / release also work on any claimed line without its token.
//   Every error is {error: "<code>", message: "<one sentence a reader can act on>"}.
//
// Storage: one Upstash Redis hash over the REST API with plain fetch, no packages.
//   readers:claims                  field "07" -> JSON {name, sent, at, tokenHash}
//   readers:rl:<ip hash>:<window>   write counter for the rate limit
// A claim is HSETNX, so the first request wins. Its token (128 random bits) is returned once, to the claimer; only its
// SHA-256 is stored. Changing a claim is a compare-and-swap in a two-line Lua script, so two changes at once cannot
// overwrite each other or bring back a line that was just given up.
//
// Env: UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN, or the Vercel Marketplace names KV_REST_API_URL +
// KV_REST_API_TOKEN. With neither: a local stand-in (a Map kept in a temp file) when VERCEL_ENV is not "production",
// so vercel dev works, and 503 "Sign-ups are not open yet" in production. READERS_ADMIN_KEY: the organiser's key.

import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { closeSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const HASH = 'readers:claims';
const RATE = { limit: 20, window: 600 };   // writes per client IP (IPv6: per /64) per 10 minutes
const MAX_NAME = 60;
const MAX_PER_PERSON = 3;   // lines one name may hold (the organiser's key is not limited)
const MAX_BODY = 4096;
// The script's line numbers. vercel.json bundles this file into the function (functions."api/readers.mjs".includeFiles).
const LINES_FILE = new URL('../data/readers-lines.json', import.meta.url);

// Compare-and-swap one field: if it still holds ARGV[2], write ARGV[3] (or delete it when ARGV[3] is empty).
const SWAP = "if redis.call('HGET', KEYS[1], ARGV[1]) ~= ARGV[2] then return 0 end " +
  "if ARGV[3] == '' then redis.call('HDEL', KEYS[1], ARGV[1]) else redis.call('HSET', KEYS[1], ARGV[1], ARGV[3]) end " +
  'return 1';

class HttpError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}
class StoreError extends Error {}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  try {
    if (!['GET', 'HEAD', 'POST'].includes(req.method)) {
      res.setHeader('Allow', 'GET, HEAD, POST');
      throw new HttpError(405, 'method', 'Use GET to read the sheet or POST to change it.');
    }
    const store = openStore(process.env);
    if (!store) throw new HttpError(503, 'not_configured', 'Sign-ups are not open yet.');
    if (req.method === 'HEAD') return reply(res, 200, {});
    if (req.method === 'GET') return reply(res, 200, sheet(await store.all(), store));
    return await change(req, res, store);
  } catch (e) {
    if (e instanceof HttpError) return reply(res, e.status, oops(e.code, e.message));
    console.error('readers:', e instanceof StoreError ? e.message : e);   // never logs tokens, names or keys
    if (e instanceof StoreError) return reply(res, 503, oops('store_unavailable', 'The sign-up sheet is not answering. Try again in a moment.'));
    return reply(res, 500, oops('server_error', 'Something went wrong on our side. Try again in a moment.'));
  }
}

async function change(req, res, store) {
  if (!/^application\/json\b/i.test(String(req.headers['content-type'] || ''))) {
    throw new HttpError(415, 'json_only', 'Send the change as JSON.');
  }
  const body = await readJson(req);
  const action = typeof body.action === 'string' ? body.action : '';
  const admin = adminCheck(req);

  // The organiser's own changes are not counted; everyone else's, and every wrong key, are.
  if (admin !== 'ok') await rateLimit(req, res, store);

  if (action === 'verify') {
    if (admin !== 'ok') refuseAdmin(admin);
    return reply(res, 200, { ok: true });
  }
  if (!['claim', 'sent', 'unsent', 'release', 'rename', 'clear'].includes(action)) {
    throw new HttpError(400, 'bad_action', 'That is not something this sheet can do.');
  }
  if ((action === 'rename' || action === 'clear') && admin !== 'ok') refuseAdmin(admin);

  const n = await lineOf(body.n, req);
  const name = action === 'claim' || action === 'rename' ? cleanName(body.name) : null;

  if (action === 'claim') {
    if (admin !== 'ok') {   // counted, not locked: two claims in the same instant could land a fourth, which is fine here
      const all = await store.all(), key = name.toLowerCase();
      const held = Object.values(all).map(readRecord).filter(c => c && c.name.toLowerCase() === key).length;
      if (held >= MAX_PER_PERSON) {
        return reply(res, 409, { ...oops('limit', `${name} already has ${MAX_PER_PERSON} lines, the most one person can take.`), ...sheet(all, store) });
      }
    }
    const token = randomBytes(16).toString('base64url');
    const rec = { name, sent: false, at: new Date().toISOString(), tokenHash: sha256(token) };
    const r = await store.claim(n, JSON.stringify(rec));
    if (!r.ok) {
      return reply(res, 409, { ...oops('taken', `Line ${n} was just taken.`), claim: publicClaim(r.all[n]), ...sheet(r.all, store) });
    }
    return reply(res, 200, { ok: true, n, token, at: rec.at, ...sheet(r.all, store) });
  }

  const raw = await store.get(n);
  const rec = readRecord(raw);
  if (!rec) {
    return reply(res, 404, { ...oops('not_claimed', `Nobody has line ${n} right now.`), ...sheet(await store.all(), store) });
  }
  if (admin !== 'ok' && !tokenMatches(body.token, rec.tokenHash)) {
    if (admin !== 'none') refuseAdmin(admin);   // a wrong key, or admin switched off: say that, not "not yours"
    throw new HttpError(403, 'not_yours', `Line ${n} was taken in another browser. Only that browser, or the organiser, can change it.`);
  }

  let next = null;   // null gives the line back
  if (action === 'sent' || action === 'unsent') next = { ...rec, sent: action === 'sent' };
  if (action === 'rename') next = { ...rec, name };
  const r = await store.swap(n, raw, next && JSON.stringify(next));
  if (!r.ok) {
    return reply(res, 409, { ...oops('changed', `Line ${n} changed while you were saving. Have another look and try again.`), ...sheet(r.all, store) });
  }
  return reply(res, 200, { ok: true, ...sheet(r.all, store) });
}

// ---------- storage ----------

function openStore(env) {
  if (env.UPSTASH_REDIS_REST_URL && env.UPSTASH_REDIS_REST_TOKEN) return upstash(env.UPSTASH_REDIS_REST_URL, env.UPSTASH_REDIS_REST_TOKEN);
  if (env.KV_REST_API_URL && env.KV_REST_API_TOKEN) return upstash(env.KV_REST_API_URL, env.KV_REST_API_TOKEN);
  if (env.VERCEL_ENV !== 'production') return memoryStore;
  return null;
}

function upstash(url, token) {
  const base = String(url).replace(/\/+$/, '');
  async function call(path, body) {
    let r, data = null;
    try {
      r = await fetch(base + path, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(5000),
      });
      data = await r.json().catch(() => null);
    } catch (e) {
      throw new StoreError(`upstash unreachable: ${e && e.name}`);
    }
    if (!r.ok || data == null) throw new StoreError(`upstash ${r.status}: ${(data && data.error) || 'no body'}`);
    return data;
  }
  const one = async cmd => {
    const d = await call('', cmd);
    if (d.error) throw new StoreError(`upstash: ${d.error}`);
    return d.result;
  };
  const pipeline = async cmds => {   // one round trip; results in order (not atomic, and does not need to be)
    const d = await call('/pipeline', cmds);
    if (!Array.isArray(d)) throw new StoreError('upstash: pipeline answer was not a list');
    return d.map(x => { if (!x || x.error) throw new StoreError(`upstash: ${(x && x.error) || 'empty'}`); return x.result; });
  };
  return {
    kind: 'redis',
    all: async () => pairs(await one(['HGETALL', HASH])),
    get: n => one(['HGET', HASH, n]),
    claim: async (n, raw) => {
      const [set, all] = await pipeline([['HSETNX', HASH, n, raw], ['HGETALL', HASH]]);
      return { ok: Number(set) === 1, all: pairs(all) };
    },
    swap: async (n, from, to) => {
      const [done, all] = await pipeline([['EVAL', SWAP, '1', HASH, n, from, to || ''], ['HGETALL', HASH]]);
      return { ok: Number(done) === 1, all: pairs(all) };
    },
    hit: async (key, ttl) => {
      const [count] = await pipeline([['INCR', key], ['EXPIRE', key, String(ttl)]]);
      return Number(count);
    },
  };
}

// HGETALL over REST is a flat [field, value, field, value, ...] list; accept an object too.
function pairs(result) {
  const out = {};
  if (Array.isArray(result)) for (let i = 0; i + 1 < result.length; i += 2) out[result[i]] = result[i + 1];
  else if (result && typeof result === 'object') Object.assign(out, result);
  return out;
}

// For vercel dev and previews without a store: an in-memory Map of the same fields. vercel dev runs every request in
// a new child process and kills it after the response, so the Map is kept in a JSON file in the temp directory (read
// fresh each request, written under a lock file so two requests at once still get first come, first served).
// On a Vercel preview that file is the instance's own /tmp: fine for trying the page, never the real sheet.
const DEV_FILE = join(tmpdir(), 'arrival-readers-dev-claims.json');
const napCell = new Int32Array(new SharedArrayBuffer(4));
function devLoad() {
  let d = {};
  try { d = JSON.parse(readFileSync(DEV_FILE, 'utf8')); } catch { /* none yet */ }
  return { hash: new Map(Object.entries(d.hash || {})), hits: new Map(Object.entries(d.hits || {})) };
}
function devChange(fn) {   // synchronous from lock to unlock, so it is atomic in this process too
  const lock = DEV_FILE + '.lock';
  let locked = false;
  for (let i = 0; i < 400 && !locked; i++) {
    try { closeSync(openSync(lock, 'wx')); locked = true; }
    catch (e) {
      if (e.code !== 'EEXIST') break;
      try { if (Date.now() - statSync(lock).mtimeMs > 3000) unlinkSync(lock); } catch { /* gone already */ }
      Atomics.wait(napCell, 0, 0, 5);
    }
  }
  try {
    const m = devLoad();
    const out = fn(m);
    const now = Date.now();
    for (const [k, h] of m.hits) if (h.until <= now) m.hits.delete(k);
    const tmp = `${DEV_FILE}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify({ hash: Object.fromEntries(m.hash), hits: Object.fromEntries(m.hits) }));
    renameSync(tmp, DEV_FILE);
    return out;
  } finally {
    if (locked) try { unlinkSync(lock); } catch { /* fine */ }
  }
}
const memoryStore = {
  kind: 'memory',
  all: async () => Object.fromEntries(devLoad().hash),
  get: async n => devLoad().hash.get(n) ?? null,
  claim: async (n, raw) => devChange(m => {
    const ok = !m.hash.has(n);
    if (ok) m.hash.set(n, raw);
    return { ok, all: Object.fromEntries(m.hash) };
  }),
  swap: async (n, from, to) => devChange(m => {
    const ok = m.hash.get(n) === from;
    if (ok) { if (to) m.hash.set(n, to); else m.hash.delete(n); }
    return { ok, all: Object.fromEntries(m.hash) };
  }),
  hit: async (key, ttl) => devChange(m => {
    const now = Date.now();
    let h = m.hits.get(key);
    if (!h || h.until <= now) m.hits.set(key, (h = { count: 0, until: now + ttl * 1000 }));
    return ++h.count;
  }),
};

// ---------- what the page sees ----------

function readRecord(raw) {
  if (typeof raw !== 'string') return null;
  try {
    const c = JSON.parse(raw);
    return c && typeof c === 'object' && typeof c.name === 'string' && c.name ? c : null;
  } catch { return null; }
}
function publicClaim(raw) {
  const c = readRecord(raw);
  return c ? { name: c.name, sent: c.sent === true, at: typeof c.at === 'string' ? c.at : null } : null;
}
function sheet(all, store) {
  const claims = {};
  for (const [n, raw] of Object.entries(all)) {
    if (!/^\d{2,3}$/.test(n)) continue;
    const c = publicClaim(raw);
    if (c) claims[n] = c;
  }
  return store.kind === 'memory' ? { claims, store: 'memory' } : { claims };
}

// ---------- checks ----------

let linesCache = null;
async function lineNumbers(req) {
  if (linesCache) return linesCache;
  let text;
  try {
    text = await readFile(LINES_FILE, 'utf8');
  } catch {
    // Not bundled (or a dev server that runs a copy of this file): read the deployment's own static copy instead.
    const host = String(req.headers['x-forwarded-host'] || req.headers.host || '').split(',')[0].trim();
    if (!/^[a-z0-9.-]+(:\d+)?$/i.test(host)) throw new Error('readers-lines.json: no file and no host to fetch it from');
    const local = /^(localhost|127\.0\.0\.1)(:\d+)?$/i.test(host);
    const proto = String(req.headers['x-forwarded-proto'] || '').split(',')[0].trim() || (local ? 'http' : 'https');
    const r = await fetch(`${proto}://${host}/data/readers-lines.json`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error(`readers-lines.json: HTTP ${r.status}`);
    text = await r.text();
  }
  const set = new Set((JSON.parse(text).lines || []).map(l => String(l.n)));
  if (!set.size) throw new Error('readers-lines.json has no lines');
  return (linesCache = set);
}

async function lineOf(v, req) {
  const s = typeof v === 'number' && Number.isInteger(v) && v >= 0 ? String(v) : typeof v === 'string' ? v.trim() : '';
  const n = /^\d{1,3}$/.test(s) ? s.padStart(2, '0') : null;
  if (!n || !(await lineNumbers(req)).has(n)) throw new HttpError(400, 'bad_line', 'That line is not in the script.');
  return n;
}

// Trimmed, 1 to 60 characters, no control characters (or the invisible marks that flip text direction).
function cleanName(v) {
  const s = typeof v === 'string' ? v.normalize('NFC').trim() : '';
  if (/[\p{Cc}\u2028\u2029\u202A-\u202E\u2066-\u2069]/u.test(s)) {
    throw new HttpError(400, 'bad_name', 'Use only letters, spaces and ordinary punctuation in the name.');
  }
  const name = s.replace(/\s+/gu, ' ');
  const len = [...name].length;
  if (len < 1) throw new HttpError(400, 'bad_name', 'Type a name first.');
  if (len > MAX_NAME) throw new HttpError(400, 'bad_name', `Keep the name to ${MAX_NAME} characters.`);
  return name;
}

function adminCheck(req) {
  const given = req.headers['x-admin-key'];
  if (typeof given !== 'string' || !given) return 'none';
  const want = process.env.READERS_ADMIN_KEY;
  if (!want) return 'off';
  return sameHex(sha256(given), sha256(want)) ? 'ok' : 'bad';
}
function refuseAdmin(admin) {
  if (admin === 'off') throw new HttpError(403, 'admin_off', 'Organiser changes are switched off: READERS_ADMIN_KEY is not set on the server.');
  if (admin === 'bad') throw new HttpError(403, 'bad_admin_key', 'That admin key is not right.');
  throw new HttpError(403, 'admin_only', 'Only the organiser can do that.');
}

function tokenMatches(token, hash) {
  return typeof token === 'string' && /^[\w-]{16,64}$/.test(token) && typeof hash === 'string' && sameHex(sha256(token), hash);
}

async function rateLimit(req, res, store) {
  const now = Math.floor(Date.now() / 1000);
  const win = Math.floor(now / RATE.window);
  const count = await store.hit(`readers:rl:${sha256(clientBucket(req)).slice(0, 24)}:${win}`, RATE.window + 60);
  if (count > RATE.limit) {
    const wait = Math.max(1, (win + 1) * RATE.window - now);
    const mins = Math.ceil(wait / 60);
    res.setHeader('Retry-After', String(wait));
    throw new HttpError(429, 'rate_limited', `That is a lot of changes from one connection. Try again in ${mins === 1 ? 'a minute' : `${mins} minutes`}.`);
  }
}

// Vercel sets x-forwarded-for itself (it does not pass on a client's own), so its first entry is the client.
function clientBucket(req) {
  const ip = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim()
    || String(req.headers['x-real-ip'] || '').trim()
    || (req.socket && req.socket.remoteAddress) || 'unknown';
  return ip.includes(':') ? ipv6Net(ip) : ip;
}
// One home or phone gets a whole IPv6 /64, so count the /64 rather than the address.
function ipv6Net(ip) {
  const a = ip.replace(/^\[|\]$/g, '').split('%')[0].toLowerCase();
  const v4 = a.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
  if (v4) return v4[1];
  const [head, tail] = a.split('::');
  const h = head ? head.split(':') : [], t = tail ? tail.split(':') : [];
  const groups = tail !== undefined ? [...h, ...Array(Math.max(0, 8 - h.length - t.length)).fill('0'), ...t] : h;
  return groups.slice(0, 4).map(g => g.replace(/^0+(?=.)/, '')).join(':') + '::/64';
}

// ---------- plumbing ----------

async function readJson(req) {
  let body;
  if ('body' in req) {   // Vercel's Node helpers have already read and parsed it (and throw on bad JSON)
    try { body = req.body; } catch { throw new HttpError(400, 'bad_json', 'That request was not valid JSON.'); }
    if (Buffer.isBuffer(body)) body = body.toString('utf8');
    if (typeof body === 'string') body = parseJson(body);
  } else {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > MAX_BODY) throw new HttpError(413, 'too_large', 'That request is too large.');
      chunks.push(chunk);
    }
    body = parseJson(Buffer.concat(chunks).toString('utf8'));
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new HttpError(400, 'bad_json', 'Send a JSON object.');
  return body;
}
function parseJson(text) {
  try { return JSON.parse(text); } catch { throw new HttpError(400, 'bad_json', 'That request was not valid JSON.'); }
}

function sha256(s) { return createHash('sha256').update(String(s)).digest('hex'); }
function sameHex(a, b) {
  const x = Buffer.from(String(a), 'hex'), y = Buffer.from(String(b), 'hex');
  return x.length === 32 && y.length === 32 && timingSafeEqual(x, y);
}
function oops(code, message) { return { error: code, message }; }
function reply(res, status, body) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(body));
}
