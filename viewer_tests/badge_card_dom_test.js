// Trainer card / badge tracker in the Splits tab.
//
// This exists because the card was deleted wholesale on 2026-08-16 (server
// aggregation and page markup both), taking the badge-consensus fix with it,
// and nothing failed to notice. A DOM test makes its removal loud.
//
// The behaviour that matters: badges render in the in-game order with
// unearned ones visibly dim (so an impossible set like Rainbow-without-Thunder
// reads as wrong at a glance), and partial worker agreement is stated rather
// than smoothed over.
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML;
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const BADGE_NAMES = ['Boulder', 'Cascade', 'Thunder', 'Rainbow',
                     'Soul', 'Marsh', 'Volcano', 'Earth'];

function card(flags, agreement, workers = 96) {
  return {
    name: 'SVER', money: 9054, badge_flags: flags,
    badge_count: flags.toString(2).split('').filter((c) => c === '1').length,
    badge_agreement: agreement,
    badges: BADGE_NAMES.map((n, i) => ({ name: n, earned: Boolean(flags & (1 << i)) })),
    play_time_seconds: 28990, play_time_maxed: false, workers,
  };
}

function splits(playerCard) {
  return {
    total_phases: 296, phases_cleared: 66, timed_phases: 59,
    sum_of_bests: 29596, sum_of_medians: 40000,
    player_card: playerCard,
    rows: [{ phase: 0, name: 'reach_viridian', clears: 12, best: 120,
             last: 380, median: 300, best_run_step: 120 }],
  };
}

let payload = splits(card(3, 1.0));

(async () => {
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

  const open = async () => {
    w.document.querySelector('[data-tab="splits"]')
      .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 300));
  };
  // The page refetches on its own interval, so waiting a fixed delay after
  // swapping the payload is a race. Wait for the DOM to actually reflect it.
  const waitFor = async (predicate, label) => {
    for (let i = 0; i < 60; i += 1) {
      await open();
      if (predicate()) return true;
      await new Promise((r) => setTimeout(r, 250));
    }
    console.log(`  [warn] timed out waiting for ${label}`);
    return false;
  };
  await open();

  const box = w.document.getElementById('playerCard');
  ok('trainer card exists and is shown', !!box && box.hidden === false);
  ok('trainer name rendered', w.document.getElementById('tcardName').textContent === 'SVER');
  ok('money formatted', /9,054/.test(w.document.getElementById('tcardMoney').textContent),
    w.document.getElementById('tcardMoney').textContent);
  ok('play time formatted as h:mm',
    /^8:03/.test(w.document.getElementById('tcardTime').textContent),
    w.document.getElementById('tcardTime').textContent);

  const chips = w.document.querySelectorAll('#tcardBadges .tcard-badge');
  ok('all eight badges rendered, earned or not', chips.length === 8, `${chips.length}`);
  const earned = [...chips].filter((c) => c.className.includes('earned'));
  ok('only earned badges are lit', earned.length === 2, `${earned.length} lit`);
  ok('count shows earned of total',
    w.document.getElementById('tcardBadgeCount').textContent === '2/8',
    w.document.getElementById('tcardBadgeCount').textContent);
  ok('badge order follows the in-game card',
    chips[0].title.startsWith('Boulder') && chips[2].title.startsWith('Thunder')
    && chips[7].title.startsWith('Earth'), chips[2].title);
  ok('unearned badge says so in its tooltip', /not yet/.test(chips[2].title), chips[2].title);
  ok('full agreement reads as settled',
    /all agreeing/.test(w.document.getElementById('tcardNote').textContent),
    w.document.getElementById('tcardNote').textContent);

  // The live bug: 62 of 96 workers correct, 34 mid-transition. The card must
  // show the majority AND say the swarm disagrees.
  payload = splits(card(3, 62 / 96));
  // Must match the partial-agreement wording specifically -- "all agreeing"
  // also contains "agree", which would satisfy the wait before the refetch.
  await waitFor(() => /% agree/.test(
    w.document.getElementById('tcardNote').textContent), 'partial agreement');
  const note = w.document.getElementById('tcardNote');
  ok('partial agreement is surfaced, not hidden', /only 65% agree/.test(note.textContent),
    note.textContent);
  ok('disagreement is styled as a warning', note.className.includes('warn'), note.className);
  ok('still shows the majority badge set while disagreeing',
    w.document.getElementById('tcardBadgeCount').textContent === '2/8');

  // A real 5-badge state must still render (the fix must not cap badges).
  payload = splits(card(31, 1.0));
  await waitFor(() =>
    w.document.getElementById('tcardBadgeCount').textContent === '5/8', 'five badges');
  ok('a genuine five-badge run renders all five',
    w.document.getElementById('tcardBadgeCount').textContent === '5/8',
    w.document.getElementById('tcardBadgeCount').textContent);

  // No card in the payload -> hidden, not a card full of zeroes.
  payload = splits(null);
  await waitFor(() =>
    w.document.getElementById('playerCard').hidden === true, 'card hidden');
  ok('card hides when the payload has none, rather than showing zeroes',
    w.document.getElementById('playerCard').hidden === true);

  console.log(fails.length ? `\n${fails.length} FAILED` : '\nall badge card checks passed');
  process.exit(fails.length ? 1 : 0);
})().catch((e) => { console.error('harness error:', e); process.exit(2); });
