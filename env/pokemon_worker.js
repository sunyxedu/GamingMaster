// Pokémon Showdown battles for env/vec_pokemon.py: one worker process hosts many battles
// (Showdown's own simulator, `sim/battle`, at the commit of the vocabulary:
// data_processing/games/pokemon/showdown_vocab.json).  JSON lines on stdin / stdout; every
// input line is {"op": ..., ...} or {"op": "batch", "cmds": [...]} and gets one output line.
//
//   start    {id, format, seed: [4 ints], teams: [p1, p2]}   packed teams, or null for the
//            format's generator (random battles)
//   choose   {id, choices: [c1, c2]}  a side's Showdown choice ("move 2 terastallize",
//            "switch 3", "team 312456") or null (no request / waiting)
//   end      {id}
//   validate {format, team}  a teambuilder export or packed team → {packed, problems}
//
// start / choose answer {id, log: [new lines], requests: [r1, r2], ended, winner: 0 | 1 |
// null, error}: the omniscient sim protocol (|split| blocks: the secret line, then the public
// one — the converter reads the public one), and each side's pending request (null when none,
// {"wait": true} while waiting for the other side).
//
//   GM_SHOWDOWN_DIR (default ~/gm-tools/pokemon-showdown): a built checkout (`node build`).
'use strict';
const path = require('path');
const os = require('os');
const readline = require('readline');

const DIR = process.env.GM_SHOWDOWN_DIR || path.join(os.homedir(), 'gm-tools', 'pokemon-showdown');
const {Battle} = require(path.join(DIR, 'dist', 'sim', 'battle'));
const {Teams} = require(path.join(DIR, 'dist', 'sim', 'teams'));
const {TeamValidator} = require(path.join(DIR, 'dist', 'sim', 'team-validator'));

const battles = new Map();   // id → {battle, cursor}

function requests(battle) {
  return battle.sides.map(s => (s.activeRequest ? s.activeRequest : null));
}

function delta(st) {
  const lines = st.battle.log.slice(st.cursor);
  st.cursor = st.battle.log.length;
  return lines;
}

function result(id, st, error) {
  const b = st.battle;
  let winner = null;
  if (b.ended) {
    if (b.winner === b.sides[0].name) winner = 0;
    else if (b.winner === b.sides[1].name) winner = 1;
  }
  return {id, log: delta(st), requests: b.ended ? [null, null] : requests(b), ended: !!b.ended, winner,
          turn: b.turn, error: error || null};
}

function start(c) {
  const opts = {formatid: c.format, seed: c.seed && c.seed.length === 4 ? c.seed : undefined};
  const battle = new Battle(opts);
  const teams = c.teams || [null, null];
  // a generated (random battle) team draws from its player's own seed — Showdown rolls a
  // fresh one when none is given, so derive both from the battle's for reproducible battles
  const s = opts.seed || [1, 2, 3, 4];
  const pseed = k => s.map((x, i) => ((x * 31 + k * 7919 + i * 104729) & 0xffff)).join(',');
  battle.setPlayer('p1', teams[0] ? {name: 'p1', team: teams[0]} : {name: 'p1', seed: pseed(1)});
  battle.setPlayer('p2', teams[1] ? {name: 'p2', team: teams[1]} : {name: 'p2', seed: pseed(2)});
  const st = {battle, cursor: 0};
  battles.set(c.id, st);
  return result(c.id, st);
}

function choose(c) {
  const st = battles.get(c.id);
  if (!st) return {id: c.id, error: `no battle ${c.id}`};
  const b = st.battle;
  let error = null;
  for (let i = 0; i < 2; i++) {
    const ch = c.choices ? c.choices[i] : null;
    if (ch === null || ch === undefined || b.ended) continue;
    const side = b.sides[i];
    if (!side.activeRequest || side.activeRequest.wait) {
      error = `p${i + 1}: a choice without a request (${ch})`;
      continue;
    }
    const ok = b.choose(side.id, ch);
    if (!ok) error = `p${i + 1}: Showdown refused "${ch}": ${side.choice && side.choice.error || ''}`;
  }
  return result(c.id, st, error);
}

function validate(c) {
  let team = c.team;
  if (team.includes('\n') || team.includes('@') || team.includes('- ')) {
    const imported = Teams.import(team);
    if (!imported) return {packed: null, problems: ['import failed']};
    team = Teams.pack(imported);
  }
  const problems = TeamValidator.get(c.format).validateTeam(Teams.unpack(team));
  return {packed: team, problems: problems || null};
}

function handle(c) {
  try {
    switch (c.op) {
      case 'start': return start(c);
      case 'choose': return choose(c);
      case 'end': battles.delete(c.id); return {id: c.id};
      case 'validate': return validate(c);
      case 'batch': return {results: c.cmds.map(handle)};
      case 'ping': return {pong: true, battles: battles.size};
      default: return {error: `unknown op ${c.op}`};
    }
  } catch (e) {
    return {id: c.id, error: String(e && e.stack || e)};
  }
}

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
rl.on('line', line => {
  if (!line.trim()) return;
  let out;
  try {
    out = handle(JSON.parse(line));
  } catch (e) {
    out = {error: String(e && e.stack || e)};
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
