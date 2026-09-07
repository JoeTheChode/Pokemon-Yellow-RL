// Sidebar splits list: must fit the sidebar width, no left/right scrolling.
//
// The list itself is the wanted design (a plain scrolling list of objectives).
// The defect was that long objective names widened the table and forced a
// horizontal scrollbar. These checks pin the fix: fixed table layout, pinned
// number columns, an ellipsized name column, and overflow-x hidden -- so no
// name length can ever make the table scroll sideways.
const { JSDOM } = require('jsdom');
const HTML = process.env.VIEWER_HTML;
const fails = [];
const ok = (l, c, d = '') => {
  console.log(`  [${c ? 'PASS' : 'FAIL'}] ${l}${d ? ' -- ' + d : ''}`);
  if (!c) fails.push(l);
};

const SPLITS = {
  total_phases: 259, phases_cleared: 5, timed_phases: 2,
  sum_of_bests: 330, sum_of_medians: 540,
  rows: [
    { phase: 0, name: 'reach_viridian', clears: 12, best: 120, last: 380, median: 300, best_run_step: 120 },
    { phase: 1, name: 'enter_viridian_mart', clears: 9, best: 210, last: 210, median: 240, best_run_step: 330 },
    { phase: 2, name: 'collect_oaks_parcel', clears: 7, best: null, last: null, median: null, best_run_step: 400 },
    { phase: 3, name: 'return_to_pallet', clears: 5, best: null, last: null, median: null, best_run_step: 700 },
    // The width stress case: the longest real objective names in the chain.
    { phase: 220, name: 'train_pikachu_mansion_level_51', clears: 2, best: 16384, last: 16384, median: 16384, best_run_step: 16384 },
  ],
};

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

  const scroll = w.document.getElementById('splitsScroll');
  const table = w.document.querySelector('.splits-table');
  const rows = w.document.querySelectorAll('#splitsBody tr');

  ok('splits list renders in the sidebar', !!scroll && rows.length === SPLITS.rows.length,
    `${rows.length} rows`);

  // The actual ask: no left/right scrolling.
  const sc = w.getComputedStyle(scroll);
  ok('no horizontal scrollbar', sc.overflowX === 'hidden', sc.overflowX);
  ok('still scrolls vertically', sc.overflowY === 'auto', sc.overflowY);

  const ts = w.getComputedStyle(table);
  ok('fixed table layout so names cannot widen it', ts.tableLayout === 'fixed', ts.tableLayout);
  ok('table is width-constrained to the sidebar', ts.width === '100%', ts.width);

  // Four pinned number columns since the elapsed-time work added Time:
  // Time / delta / Last / Best. What this check really pins is that every
  // metric column has a fixed width and the *name* column is the only
  // flexible one -- that is what stops a long objective name widening the
  // table. So assert the shape, not just the count.
  const cols = [...w.document.querySelectorAll('.splits-table colgroup col')];
  ok('every number column pinned, name column flexible', cols.length === 5
    && !cols[0].getAttribute('style')
    && cols.slice(1).every((c) => /width/.test(c.getAttribute('style') || '')),
    `${cols.length} cols`);

  const nameCell = rows[4].querySelector('td');
  const ncs = w.getComputedStyle(nameCell);
  ok('long name ellipsizes rather than widening', ncs.textOverflow === 'ellipsis', ncs.textOverflow);
  ok('long name kept on one line', ncs.whiteSpace === 'nowrap', ncs.whiteSpace);
  // The page humanizes names for display (underscores -> spaces), so assert
  // the rendered form rather than the raw objective id.
  ok('full name available on hover',
    /train.pikachu.mansion.level.51/.test(nameCell.getAttribute('title') || ''),
    nameCell.getAttribute('title'));

  // Content correctness still holds.
  ok('best and delta rendered', /120/.test(rows[0].textContent) && /\+260/.test(rows[0].textContent),
    rows[0].textContent.replace(/\s+/g, ' ').trim());
  ok('untimed objective shows dashes not zeros',
    rows[2].querySelectorAll('td')[3].textContent.trim() === '—');
  ok('summary totals shown', w.document.getElementById('splitsSumBest').textContent === '330');

  w.__applyAgents([{ env_id: 0, user: 'S', color: '#fff', last_position: [1, 2, 3], map_id: 3, quest_phase: 3 }]);
  await new Promise((r) => setTimeout(r, 150));
  w.document.querySelector('[data-tab="splits"]').dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 250));
  const cur = w.document.querySelectorAll('#splitsBody tr.current');
  ok('frontier objective highlighted', cur.length === 1 && /return.to.pallet/.test(cur[0].textContent),
    `${cur.length} highlighted`);

  console.log(fails.length ? `\n${fails.length} FAILED` : '\nall splits list checks passed');
  process.exit(fails.length ? 1 : 0);
})().catch((e) => { console.error('harness error:', e); process.exit(2); });
