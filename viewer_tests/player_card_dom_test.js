// Splits tab: percent-complete readout and the in-game-style trainer card.
//
// The card answers "how far through the game are we" at a glance. Two things
// it must never do: imply the percentage is time-weighted (about a third of
// the 296 objectives are level grinds), and render a card of zeroes when the
// trainer is still on a wrapper that does not publish the fields yet.
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML;
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const ROWS = [
  { phase: 0, name: 'reach_viridian', clears: 12, best: 120, last: 380, median: 300, best_run_step: 120 },
  { phase: 1, name: 'beat_brock', clears: 9, best: 369, last: 429, median: 400, best_run_step: 330 },
];

// Boulder + Cascade earned (bits 0 and 1), 39 of 296 objectives cleared.
const WITH_CARD = {
  total_phases: 296, phases_cleared: 39, timed_phases: 2,
  percent_complete: 13.2, sum_of_bests: 18694, sum_of_medians: 26025,
  rows: ROWS,
  player_card: {
    name: 'PIKA', money: 3740, badge_flags: 0b00000011, badge_count: 2,
    badges: [
      { name: 'Boulder', earned: true }, { name: 'Cascade', earned: true },
      { name: 'Thunder', earned: false }, { name: 'Rainbow', earned: false },
      { name: 'Soul', earned: false }, { name: 'Marsh', earned: false },
      { name: 'Volcano', earned: false }, { name: 'Earth', earned: false },
    ],
    play_time_seconds: 3 * 3600 + 7 * 60 + 42, play_time_maxed: false, workers: 96,
  },
};

// A trainer still running the old wrapper publishes no card fields at all.
const NO_CARD = {
  total_phases: 296, phases_cleared: 39, timed_phases: 2,
  percent_complete: 13.2, sum_of_bests: 18694, sum_of_medians: 26025,
  rows: ROWS, player_card: null,
};

async function render(payload) {
  const dom = await JSDOM.fromFile(HTML, {
    runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.EventSource = undefined;
      w.fetch = async (u) => {
        const s = String(u);
        if (s.includes('quest_splits.json')) return { ok: true, json: async () => payload };
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
  const w = await render(WITH_CARD);
  const doc = w.document;
  const card = doc.getElementById('playerCard');

  ok('trainer card is shown when the trainer publishes the fields', card && !card.hidden);
  ok('percent complete appears in the splits header',
    /13\.2%/.test(doc.getElementById('splitsStatus').textContent),
    doc.getElementById('splitsStatus').textContent);
  ok('percent complete appears on the card',
    doc.getElementById('tcardPct').textContent === '13.2%',
    doc.getElementById('tcardPct').textContent);

  ok('player name rendered', doc.getElementById('tcardName').textContent === 'PIKA',
    doc.getElementById('tcardName').textContent);
  ok('money rendered with a separator',
    /3,740/.test(doc.getElementById('tcardMoney').textContent),
    doc.getElementById('tcardMoney').textContent);
  // 3h07m42s -> "3:07"; seconds are noise on a multi-hour run.
  ok('play time rendered as h:mm', doc.getElementById('tcardTime').textContent === '3:07',
    doc.getElementById('tcardTime').textContent);

  const badges = doc.querySelectorAll('#tcardBadges .tcard-badge');
  const earned = doc.querySelectorAll('#tcardBadges .tcard-badge.earned');
  ok('eight badge slots always drawn', badges.length === 8, `${badges.length}`);
  ok('only earned badges are lit', earned.length === 2, `${earned.length} lit`);
  ok('badge count matches the flags',
    doc.getElementById('tcardBadgeCount').textContent === '2/8',
    doc.getElementById('tcardBadgeCount').textContent);
  ok('unearned badge still names itself on hover',
    /Thunder/.test(badges[2].getAttribute('title') || ''),
    badges[2].getAttribute('title'));

  // The honesty check: never let the number read as "13% of the playthrough".
  const note = doc.getElementById('tcardNote').textContent;
  ok('note says the percentage counts objectives, not time',
    /objectives/i.test(note) && /not elapsed game time/i.test(note), note);

  const w2 = await render(NO_CARD);
  ok('card hidden rather than showing zeroes when unavailable',
    w2.document.getElementById('playerCard').hidden === true);
  ok('percent still shown without a card',
    /13\.2%/.test(w2.document.getElementById('splitsStatus').textContent));

  console.log();
  if (fails.length) { console.log(`${fails.length} FAILED: ${fails.join(', ')}`); process.exit(1); }
  console.log('all trainer-card checks passed');
})().catch((e) => { console.error('harness error:', e); process.exit(1); });
