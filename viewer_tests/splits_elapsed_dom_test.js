// Splits tab: the per-objective elapsed-time column, the steps/time unit
// toggle, and the time-sink ranking.
//
// The Time column is always frontier dwell and never switches units -- it is
// the only additive time figure on the page, so a toggle that silently turned
// it into per-worker attempt time would make the "biggest time sink" reading
// wrong. These checks pin that, plus the sidebar-width constraint the table
// already had before a fifth column was added to it.
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML;
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const NOW = Date.now() / 1000;
const SPLITS = {
  total_phases: 302, live_phase: 4, phases_cleared: 4, remaining: 298,
  percent_complete: 1.3, ledger_rows: 302, timed_phases: 2,
  sum_of_bests: 330, sum_of_medians: 540,
  attempt_reset_id: 'fullgame-TEST', attempt_started_ts: NOW - 7200,
  tracked_run_secs: 5400, current_dwell_secs: 900, dwelt_phases: 5,
  time_sink_share: 83.4,
  time_sinks: [
    { phase: 3, name: 'train_pikachu_level_10', dwell_secs: 3600, share: 66.7 },
    { phase: 4, name: 'enter_viridian_forest', dwell_secs: 900, share: 16.7 },
  ],
  rows: [
    { phase: 0, name: 'reach_viridian', clears: 12, best: 120, last: 380, median: 300,
      best_run_step: 120, dwell_secs: 600, visits: 1, attempt_clears: 12,
      best_secs: 45, last_secs: 90, median_secs: 60, timed_clears: 8 },
    { phase: 1, name: 'enter_viridian_mart', clears: 9, best: 210, last: 210, median: 240,
      best_run_step: 330, dwell_secs: 240, visits: 1, attempt_clears: 9, dwell_estimated: true,
      best_secs: 12, last_secs: 12, median_secs: 12, timed_clears: 5 },
    // Cleared, but never timed: must stay blank rather than render a zero.
    { phase: 2, name: 'collect_oaks_parcel', clears: 7, best: null, last: null, median: null,
      best_run_step: 400, dwell_secs: 60, visits: 1, attempt_clears: 7,
      best_secs: null, last_secs: null, median_secs: null, timed_clears: null },
    { phase: 3, name: 'train_pikachu_level_10', clears: 2, best: 16384, last: 16384, median: 16384,
      best_run_step: 16384, dwell_secs: 3600, visits: 2, attempt_clears: 2,
      best_secs: 300, last_secs: 300, median_secs: 300, timed_clears: 2 },
    // Current objective, still accruing.
    { phase: 4, name: 'enter_viridian_forest', clears: 0, best: null, last: null, median: null,
      best_run_step: null, dwell_secs: 900, visits: 1, attempt_clears: null,
      best_secs: null, last_secs: null, median_secs: null, timed_clears: null },
  ],
};

const cells = (w, rowIndex) =>
  [...w.document.querySelectorAll('#splitsBody tr')[rowIndex].querySelectorAll('td')]
    .map((td) => td.textContent.trim());

(async () => {
  const dom = await JSDOM.fromFile(HTML, {
    runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.EventSource = undefined;
      w.fetch = async (u) => {
        const s = String(u);
        if (s.includes('quest_splits.json')) return { ok: true, json: async () => SPLITS };
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

  // ------------------------------------------------------ the Time column
  const heads = [...w.document.querySelectorAll('.splits-table thead th')]
    .map((th) => th.textContent.trim());
  ok('table has Objective / Time / Δ / Last / Best',
    heads.join('|') === 'Objective|Time|Δ|Last|Best', heads.join('|'));

  ok('dwell renders in the Time column, in time units',
    cells(w, 0)[1] === '10m', cells(w, 0)[1]);
  ok('an hour of dwell reads in hours',
    cells(w, 3)[1] === '1.0h', cells(w, 3)[1]);
  ok('the current objective shows its accruing dwell',
    cells(w, 4)[1] === '15m', cells(w, 4)[1]);

  const hot = w.document.querySelectorAll('#splitsBody td.dwell.hot');
  ok('only the worst objectives are flagged hot', hot.length === 1, `${hot.length} flagged`);

  // A split estimate must be visibly distinct from a measurement, or the
  // audit reads a guess as data.
  ok('estimated dwell is prefixed and styled apart',
    cells(w, 1)[1] === '~4m'
    && w.document.querySelectorAll('#splitsBody tr')[1]
      .querySelectorAll('td')[1].classList.contains('est'),
    cells(w, 1)[1]);
  ok('measured dwell carries no estimate marker',
    cells(w, 0)[1] === '10m'
    && !w.document.querySelectorAll('#splitsBody tr')[0]
      .querySelectorAll('td')[1].classList.contains('est'),
    cells(w, 0)[1]);
  ok('the estimate is explained in the tooltip',
    /estimated/.test(w.document.querySelectorAll('#splitsBody tr')[1]
      .querySelectorAll('td')[1].title),
    w.document.querySelectorAll('#splitsBody tr')[1].querySelectorAll('td')[1].title.slice(0, 60));

  // ------------------------------------------------------- the unit toggle
  ok('unit toggle defaults to steps',
    cells(w, 0)[3] === '380' && cells(w, 0)[4] === '120',
    cells(w, 0).join('|'));

  w.document.querySelector('#splitsUnitToggle button[data-unit="secs"]')
    .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 150));

  const timed = cells(w, 0);
  ok('toggling to time switches Last/Best to per-worker attempt time',
    timed[3] === '90s' && timed[4] === '45s', timed.join('|'));
  ok('sub-minute attempt times keep second resolution rather than rounding to 2m',
    timed[3] === '90s', timed[3]);
  ok('the delta follows the same unit', timed[2] === '+45s', timed[2]);
  ok('the Time column does NOT change with the toggle -- it is frontier time',
    timed[1] === '10m', timed[1]);
  ok('an objective with no attempt clock stays blank, not zero',
    cells(w, 2)[3] === 'done' && cells(w, 2)[4] === 'done',
    cells(w, 2).join('|'));

  w.document.querySelector('#splitsUnitToggle button[data-unit="steps"]')
    .dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 150));
  ok('toggling back restores steps', cells(w, 0)[3] === '380', cells(w, 0)[3]);

  // --------------------------------------------------------- time sinks
  // A block that grows downward takes its height out of a splits table that
  // only shows a handful of rows, so the shortlist is one line and the full
  // ranking is the table sorted by Time.
  ok('no time-sink block exists to steal table height',
    !w.document.getElementById('splitsSinks')
    && !w.document.getElementById('splitsSinkRows'));
  const topSink = w.document.getElementById('splitsTopSink');
  ok('the worst objective is summarised on one line',
    /train pikachu level 10/.test(topSink.textContent)
    && /1\.0h/.test(topSink.textContent) && /66\.7%/.test(topSink.textContent),
    topSink.textContent.trim());
  ok('the summary says what the shortlist covers between them',
    /1 other objective it accounts for 83\.4% of tracked time/.test(topSink.title),
    topSink.title);

  // ------------------------------------------------------------- totals
  const text = (id) => w.document.getElementById(id).textContent.trim();
  ok('attempt elapsed carries accounted-for time on the same row',
    /^2h 0m \d+s \(1\.5h tracked\)$/.test(text('splitsAttemptElapsed')),
    text('splitsAttemptElapsed'));
  ok('the elapsed/tracked gap is explained',
    /\(75%\)/.test(w.document.getElementById('splitsAttemptElapsed').title),
    w.document.getElementById('splitsAttemptElapsed').title);
  ok('time on the current objective is rendered',
    text('splitsCurrentDwell') === '15m 0s', text('splitsCurrentDwell'));

  // ---------------------------------------------------- sorting by cost
  const names = () => [...w.document.querySelectorAll('#splitsBody tr')]
    .map((tr) => tr.querySelector('.split-objective').textContent.trim());
  ok('chain order is the default', names()[0] === 'reach viridian', names()[0]);
  const sortTh = w.document.getElementById('splitsSortTime');
  sortTh.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 150));
  ok('clicking Time ranks every objective by wall-clock cost',
    names()[0] === 'train pikachu level 10'
    && names()[1] === 'enter viridian forest', names().slice(0, 3).join('|'));
  ok('the sorted column is marked active',
    sortTh.classList.contains('active') && /▾/.test(sortTh.textContent),
    sortTh.textContent.trim());
  sortTh.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 150));
  ok('clicking again restores chain order',
    names()[0] === 'reach viridian' && !sortTh.classList.contains('active'),
    names()[0]);

  // ------------------------------------- the pre-existing width constraint
  const scroll = w.document.getElementById('splitsScroll');
  const table = w.document.querySelector('.splits-table');
  ok('five columns still use a fixed layout',
    w.getComputedStyle(table).tableLayout === 'fixed',
    w.getComputedStyle(table).tableLayout);
  ok('the splits list still never scrolls sideways',
    w.getComputedStyle(scroll).overflowX === 'hidden',
    w.getComputedStyle(scroll).overflowX);

  console.log();
  if (fails.length) {
    console.log(`FAILED (${fails.length}): ${JSON.stringify(fails)}`);
    process.exit(1);
  }
  console.log('all splits elapsed-time DOM checks passed');
})();
