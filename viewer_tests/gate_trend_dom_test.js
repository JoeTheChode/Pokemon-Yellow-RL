// Gate card trend row: verdict pill, plateau-counter sparkline, detail line.
//
// The point of the row is that "0 of 20" is ambiguous, so the checks that
// matter are the ones separating the three readings of it: a counter that
// keeps resetting (improving), one that climbs (converging), and one that
// never moves (hung). Each must be visually distinct, not just worded
// differently.
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML;
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const IMPROVING = {
  phase: 37, verdict: 'improving', watched_s: 840, clears: 46,
  clears_per_min: 3.3, idle_s: 5, stall_after_s: 120, typical_clear_s: 18,
  improvements: 4, improvement_age_s: 180, improvement_delta: 87,
  best: 1432, first_best: 1600,
  history: [0, 3, 6, 0, 2, 5, 9, 0, 1, 4, 0, 2],
};
const CONVERGING = {
  phase: 37, verdict: 'converging', watched_s: 900, clears: 18,
  clears_per_min: 1.2, idle_s: 20, stall_after_s: 120, typical_clear_s: 50,
  improvements: 0, improvement_age_s: null, improvement_delta: null,
  best: 1432, first_best: 1432, history: [2, 5, 8, 11, 14, 17],
};
const STALLED = {
  phase: 37, verdict: 'stalled', watched_s: 1800, clears: 6,
  clears_per_min: null, idle_s: 640, stall_after_s: 120, typical_clear_s: 12,
  improvements: 0, improvement_age_s: null, improvement_delta: null,
  best: 1432, first_best: 1432, history: [1, 2, 3, 4, 5, 6],
};

const splits = (trend) => ({
  total_phases: 302, live_phase: 37, phases_cleared: 37, remaining: 265,
  percent_complete: 12.3, ledger_rows: 302, timed_phases: 2,
  sum_of_bests: 330, sum_of_medians: 540, rows: [
    { phase: 36, name: 'enter_mt_moon', clears: 4, best: 210, last: 210, median: 210,
      dwell_secs: 120, best_secs: 20, last_secs: 20, median_secs: 20 },
    { phase: 37, name: 'beat_mt_moon_jessie_james', clears: 46, best: 1432, last: 1500,
      median: 1480, dwell_secs: 840, best_secs: 90, last_secs: 95, median_secs: 92 },
  ],
  mastery: {
    config: { no_improve_rotations: 20, grind_successes: 5, window_size: 32 },
    frontier_phase: 37,
    frontier: {
      phase: 37, name: 'beat_mt_moon_jessie_james', mode: 'plateau',
      progress: 0, target: 20, samples: 12, allowed: false,
    },
    published: {
      phase: 37, name: 'beat_mt_moon_jessie_james', mode: 'plateau',
      progress: 0, target: 20, samples: 12, allowed: false,
    },
    live: { phase: 37, n: 96, majority: 96, name: 'beat_mt_moon_jessie_james', age_s: 0.3 },
    frontier_age_s: 900,
    trend,
  },
});

(async () => {
  const dom = await JSDOM.fromFile(HTML, {
    runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.EventSource = undefined;
      w.fetch = async (u) => String(u).includes('quest_splits.json')
        ? { ok: true, json: async () => splits(IMPROVING) }
        : { ok: false, status: 404, json: async () => ({}) };
    },
  });
  const w = dom.window;
  await new Promise((r) => w.addEventListener('load', r));
  await new Promise((r) => setTimeout(r, 500));
  w.document.querySelector('[data-tab="splits"]').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 350));

  const box = () => w.document.getElementById('gateTrend');
  const pill = () => w.document.getElementById('gateTrendPill');
  const detail = () => w.document.getElementById('gateTrendDetail');
  const spark = () => w.document.getElementById('gateTrendSpark');
  // `renderGateTrend` is a function declaration, so it is on the window and
  // each verdict can be driven directly instead of reloading the document
  // once per case. (`splitsState` is a const and deliberately is not.)
  const show = (trend) => w.renderGateTrend({ trend }, 20);

  // ------------------------------------------------------------ improving
  ok('the trend row is shown when a trend exists', !box().hidden);
  ok('the improving verdict is labelled and colour-coded',
    pill().textContent.trim() === 'improving'
    && pill().classList.contains('improving'), pill().className);
  ok('improving explains that promotion is delayed by progress, not a hang',
    /not by a hang/.test(pill().title), pill().title.slice(0, 50));
  ok('the detail line reports the best and the size of the last gain',
    /best 1,432/.test(detail().textContent)
    && /−87 steps 3m ago/.test(detail().textContent), detail().textContent);
  ok('the detail line reports how many faster routes were found',
    /4 faster routes in 14m/.test(detail().textContent), detail().textContent);
  ok('the detail tooltip reports the best route coming down',
    /come down from 1,600 to 1,432 steps/.test(detail().title), detail().title);

  const line = () => spark().querySelector('.spark-line');
  ok('a sparkline is drawn for the plateau counter',
    !!line() && (line().getAttribute('d') || '').startsWith('M'),
    (line() && line().getAttribute('d') || '').slice(0, 24));
  ok('the sparkline is colour-matched to the verdict',
    line().classList.contains('improving'), line().className.baseVal || line().className);
  ok('the promotion target is drawn as a reference line',
    !!spark().querySelector('.spark-target'));

  const improvingPath = line().getAttribute('d');

  // ----------------------------------------------------------- converging
  show(CONVERGING);
  ok('the converging verdict is labelled and colour-coded',
    pill().textContent.trim() === 'converging'
    && pill().classList.contains('converging'), pill().className);
  ok('converging says no faster route has landed',
    /no faster route in 15m/.test(detail().textContent), detail().textContent);
  ok('converging redraws a different shape from improving',
    line().getAttribute('d') !== improvingPath);
  ok('the converging sparkline recolours too',
    line().classList.contains('converging')
    && !line().classList.contains('improving'), line().className.baseVal);

  // -------------------------------------------------------------- stalled
  show(STALLED);
  ok('the stalled verdict is labelled and colour-coded',
    pill().textContent.trim() === 'stalled'
    && pill().classList.contains('stalled'), pill().className);
  ok('stalled leads with how long it has been quiet',
    /no clears in 11m/.test(detail().textContent), detail().textContent);
  ok('stalled contrasts that against the normal cadence',
    /normally every 12s/.test(detail().textContent), detail().textContent);
  ok('stalled does not claim a clear rate it no longer has',
    !/\/min/.test(detail().textContent), detail().textContent);

  // ------------------------------------------------- absent / no data yet
  show(null);
  ok('the row hides entirely when there is no trend', box().hidden);

  show({ ...IMPROVING, verdict: 'watching', clears: 0, clears_per_min: null,
    improvements: 0, improvement_age_s: null, improvement_delta: null,
    watched_s: 25, history: [] });
  ok('watching says so rather than implying a hang',
    pill().textContent.trim() === 'watching'
    && /nothing yet · watching 25s/.test(detail().textContent),
    detail().textContent);
  ok('an empty history draws no line rather than a broken path',
    !spark().querySelector('.spark-line'));

  console.log();
  if (fails.length) {
    console.log(`FAILED (${fails.length}): ${JSON.stringify(fails)}`);
    process.exit(1);
  }
  console.log('all gate-trend DOM checks passed');
})();
