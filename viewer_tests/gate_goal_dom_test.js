// Objective card: says what the current objective *is*, not just its index.
//
// The defect: the card's title came from the splits ledger, which is built
// from [QUEST] completion lines. The objective being drilled has never been
// completed, so it has no ledger row and the title fell back to the literal
// "phase 56" -- an index, with no statement of what the agents were trying to
// do. These checks pin the name coming from the payload and the clear
// condition being rendered, plus the two degraded cases (no chain exported,
// chain stale against train.py).
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML || 'pokemon_yellow_map_viewer.html';
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const frontier = (over = {}) => Object.assign({
  phase: 56,
  name: 'beat_cerulean_rival',
  kind: 'event',
  goal: 'Clears when event flag 152 (EVENT_BEAT_CERULEAN_RIVAL) is set.',
  criteria: ['event flag 152 (EVENT_BEAT_CERULEAN_RIVAL) is set'],
  mode: 'plateau',
  progress: 0, target: 20, samples: 0, window_size: 32,
  under_budget: 0, budget: 2048, hit_rate: null, needed_hits: 0,
  short_by: 20, allowed: false,
}, over);

const payload = (mastery) => ({
  total_phases: 296, phases_cleared: 56, timed_phases: 2,
  sum_of_bests: 330, sum_of_medians: 540,
  rows: [
    { phase: 55, name: 'beat_misty', clears: 317, best: 521, last: 629, median: 629, best_run_step: 487 },
  ],
  mastery,
});

async function load(mastery) {
  const dom = await JSDOM.fromFile(HTML, {
    runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.EventSource = undefined;
      w.fetch = async (u) => {
        const s = String(u);
        if (s.includes('quest_splits.json')) return { ok: true, json: async () => payload(mastery) };
        if (s.includes('map_regions') || s.includes('map_data')) return { ok: true, json: async () => ({}) };
        return { ok: false, status: 404, json: async () => ({}) };
      };
    },
  });
  const w = dom.window;
  await new Promise((r) => w.addEventListener('load', r));
  await new Promise((r) => setTimeout(r, 500));
  w.document.querySelector('[data-tab="splits"]').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 350));
  return w;
}

(async () => {
  const cfg = { window_size: 32, no_improve_rotations: 20, grind_successes: 5 };

  // 1. The normal case: named objective, clear condition spelled out.
  let w = await load({ config: cfg, frontier_phase: 56, frontier: frontier(), phases: [], chain_stale: false });
  let title = w.document.getElementById('gateTitle').textContent;
  let goalBox = w.document.getElementById('gateGoal');
  ok('title names the objective, not the phase index',
    /beat cerulean rival/.test(title) && !/^phase /.test(title), title);
  ok('card states the clear condition', !goalBox.hidden
    && /event flag 152/.test(goalBox.textContent), goalBox.textContent);
  ok('event name is resolved, not a bare bit number',
    /EVENT_BEAT_CERULEAN_RIVAL/.test(goalBox.textContent));
  ok('promotion rule still rendered',
    /20 consecutive clears/.test(w.document.getElementById('gateRule').textContent));

  // 2. A grind frontier must read as reliability, not route optimization.
  //    This is what the empty ledger name used to break: the mode is picked
  //    off the `train_` prefix, so an unnamed frontier always looked like a
  //    plateau objective and the card described the wrong promotion rule.
  w = await load({
    config: cfg, frontier_phase: 60,
    frontier: frontier({
      phase: 60, name: 'train_pikachu_level_31', kind: 'grind', mode: 'reliability',
      goal: 'Clears when the lead Pokémon is level 31+ and no battle or dialogue is on screen.',
      progress: 2, target: 5,
    }),
    phases: [], chain_stale: false,
  });
  ok('grind frontier reads as RNG reliability',
    w.document.getElementById('gateMode').textContent === 'RNG reliability',
    w.document.getElementById('gateMode').textContent);
  ok('grind clear condition names the level bar',
    /level 31\+/.test(w.document.getElementById('gateGoal').textContent));

  // 3. No chain exported: degrade to the old name-only card, no empty line.
  w = await load({
    config: cfg, frontier_phase: 56,
    frontier: frontier({ goal: '', criteria: [], kind: '' }),
    phases: [], chain_stale: null,
  });
  ok('missing chain hides the goal line rather than showing an empty one',
    w.document.getElementById('gateGoal').hidden);

  // 4. Chain stale against train.py: still shown, but flagged. A quest edit
  //    shifts every later phase index, so a stale description can name the
  //    wrong objective entirely -- silence would be the dangerous choice.
  w = await load({
    config: cfg, frontier_phase: 56, frontier: frontier(), phases: [], chain_stale: true,
  });
  goalBox = w.document.getElementById('gateGoal');
  ok('stale chain is flagged on the card', !goalBox.hidden
    && /out of date/.test(goalBox.textContent), goalBox.textContent);
  ok('stale warning is marked up for styling, not just text',
    !!goalBox.querySelector('.gate-stale'));

  console.log(fails.length ? `\n${fails.length} FAILED` : '\nall objective card checks passed');
  process.exit(fails.length ? 1 : 0);
})().catch((e) => { console.error('harness error:', e); process.exit(2); });
