// DOM test for the viewer's agent list, run against the real page in jsdom.
//
// The bug being pinned down: the ranked agent list re-sorted its DOM on every
// stream tick (rank depends on visited-tile counts, which change constantly),
// so a row moved out from under the pointer between press and release and the
// row's own `click` handler never fired. These tests drive the real page.
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

// Works both from the repo (viewer_tests/ beside the sources' parent) and
// from a scratchpad copy sitting alongside the sources. Override with
// VIEWER_HTML to point at any other build (e.g. a pre-fix baseline).
const HTML = process.env.VIEWER_HTML
  || [
    path.join(__dirname, 'pokemon_yellow_map_viewer.html'),
    path.join(__dirname, '..', 'pokemon_yellow_map_viewer.html'),
  ].find((p) => fs.existsSync(p));
const failures = [];

function check(label, ok, detail = '') {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}${detail ? ' -- ' + detail : ''}`);
  if (!ok) failures.push(label);
}

function makeAgents(n, seed = 0) {
  return Array.from({ length: n }, (_, i) => ({
    env_id: i,
    user: 'SVER-YV',
    color: '#FFD700',
    last_position: [3 + (i % 5), 4 + (i % 7), 6],
    map_id: 6,
    pikachu_level: 5,
    battle_status: 'wild',
    battle_source: 'wIsInBattle@D056',
    // The churn driver: every agent's visited count moves every tick, which
    // is what made the ranking (and therefore the DOM order) unstable.
    visited_count: 1000 + ((i * 37 + seed * 101) % 500),
    progress_count: 10 + ((i + seed) % 3),
    quest_phase: 126,
    quest_total: 156,
    last_seen: Date.now() / 1000,
  }));
}

async function main() {
  const dom = await JSDOM.fromFile(HTML, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'http://localhost/',
    beforeParse(window) {
      // The page fetches map regions, heatmap, history and opens an SSE
      // stream. Stub all of it: this test is only about list/DOM behaviour.
      window.EventSource = undefined;
      window.fetch = async (url) => {
        const u = String(url);
        if (u.includes('map_regions') || u.includes('map_data')) {
          return { ok: true, json: async () => ({}) };
        }
        return { ok: false, json: async () => ({}), status: 404 };
      };
    },
  });
  const { window } = dom;
  await new Promise((r) => window.addEventListener('load', r, { once: true }));
  // Let the page's own boot-time async work settle.
  await new Promise((r) => setTimeout(r, 300));

  const applyAgentPayload = window.applyAgentPayload;
  const markerList = window.document.getElementById('markerList');

  check('page exposes applyAgentPayload for driving', typeof applyAgentPayload === 'function');
  if (typeof applyAgentPayload !== 'function') {
    console.log('  (page keeps its functions module-scoped; falling back to event dispatch only)');
  }

  // ---- tick 1: initial render -------------------------------------------
  window.__applyAgents(makeAgents(96, 0));
  let rows = markerList.querySelectorAll('.live-agent-row[data-env-id]');
  check('96 agent rows rendered', rows.length === 96, `got ${rows.length}`);

  const orderAfterFirst = Array.from(
    markerList.querySelectorAll('.live-agent-row[data-env-id]')
  ).map((n) => n.dataset.envId);
  const firstRowKey = orderAfterFirst[0];
  const detailBefore = markerList
    .querySelector(`.live-agent-row[data-env-id="${firstRowKey}"] .live-detail`)
    .textContent;

  // ---- ticks 2..6: heavy churn, immediately after the first render ------
  // Ordering is rate-limited, so back-to-back ticks must NOT reshuffle rows.
  for (let seed = 1; seed <= 5; seed += 1) {
    window.__applyAgents(makeAgents(96, seed));
  }
  const orderAfterChurn = Array.from(
    markerList.querySelectorAll('.live-agent-row[data-env-id]')
  ).map((n) => n.dataset.envId);
  check(
    'row order is stable across rapid ticks (reorder throttled)',
    JSON.stringify(orderAfterFirst) === JSON.stringify(orderAfterChurn),
    `${orderAfterFirst.slice(0, 5)} vs ${orderAfterChurn.slice(0, 5)}`
  );

  // Content must still be live even though ordering is frozen: the same row
  // (same env id, same DOM slot) must show refreshed numbers.
  const detailAfter = markerList
    .querySelector(`.live-agent-row[data-env-id="${firstRowKey}"] .live-detail`)
    .textContent;
  check(
    'row content still updates while order is frozen',
    detailAfter !== detailBefore && /tiles/.test(detailAfter),
    `"${detailBefore.slice(0, 45)}" -> "${detailAfter.slice(0, 45)}"`
  );

  // ---- the actual regression: click a row mid-churn ----------------------
  // Simulate the real failure mode: pointer goes down on a row, the list
  // re-sorts, then the pointer comes up. Under the old per-row `click`
  // wiring nothing fired. With delegated pointerdown, focus is taken on
  // press and survives whatever the list does afterwards.
  const target = markerList.querySelector('.live-agent-row[data-env-id="42"]');
  check('row for env 42 exists', !!target);

  const down = new window.PointerEvent('pointerdown', { bubbles: true, cancelable: true });
  target.dispatchEvent(down);
  // list churns between press and release
  window.__applyAgents(makeAgents(96, 99));
  const up = new window.PointerEvent('pointerup', { bubbles: true, cancelable: true });
  target.dispatchEvent(up);

  check(
    'clicking a row focuses that agent despite mid-click churn',
    window.__focusedEnvId() === '42',
    `focusedEnvId=${window.__focusedEnvId()}`
  );

  // ---- keyboard activation still works ----------------------------------
  window.__setFocused(null);
  const row7 = markerList.querySelector('.live-agent-row[data-env-id="7"]');
  row7.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  check('Enter on a focused row activates follow', window.__focusedEnvId() === '7',
    `focusedEnvId=${window.__focusedEnvId()}`);

  // ---- hovering the list freezes reordering entirely ---------------------
  window.__setFocused(null);
  markerList.dispatchEvent(new window.PointerEvent('pointerenter', { bubbles: false }));
  const beforeHover = Array.from(
    markerList.querySelectorAll('.live-agent-row[data-env-id]')
  ).map((n) => n.dataset.envId);
  // Push well past the throttle window so only the hover guard can hold it.
  window.__advanceReorderClock(-60000);
  for (let seed = 200; seed <= 205; seed += 1) window.__applyAgents(makeAgents(96, seed));
  const duringHover = Array.from(
    markerList.querySelectorAll('.live-agent-row[data-env-id]')
  ).map((n) => n.dataset.envId);
  check('hovering the list suppresses reordering',
    JSON.stringify(beforeHover) === JSON.stringify(duringHover));

  // ---- once un-hovered and past the throttle, ranking does apply ---------
  markerList.dispatchEvent(new window.PointerEvent('pointerleave', { bubbles: false }));
  window.__advanceReorderClock(-60000);
  window.__applyAgents(makeAgents(96, 300));
  const afterUnhover = Array.from(
    markerList.querySelectorAll('.live-agent-row[data-env-id]')
  ).map((n) => n.dataset.envId);
  check('ranking still reapplies once idle and un-hovered',
    JSON.stringify(afterUnhover) !== JSON.stringify(duringHover),
    `head now ${afterUnhover.slice(0, 5)}`);

  // ---- battle dot trust gate --------------------------------------------
  const trusted = markerList.querySelector('.live-agent-row .battle-dot');
  check('trusted battle_source colours the dot',
    trusted.className.includes('battle-wild'), trusted.className);

  const untrusted = makeAgents(96, 400).map((a) => {
    const copy = { ...a };
    delete copy.battle_source; // pre-fix wrapper
    return copy;
  });
  window.__applyAgents(untrusted);
  const dot = markerList.querySelector('.live-agent-row .battle-dot');
  check('missing battle_source leaves the dot uncoloured (not a false "wild")',
    !/battle-(wild|trainer|overworld)/.test(dot.className), dot.className);
  check('untrusted dot explains itself in the tooltip',
    /unavailable/i.test(dot.title), dot.title);

  dom.window.close();
  console.log();
  if (failures.length) {
    console.log(`FAILED (${failures.length}): ${JSON.stringify(failures)}`);
    process.exit(1);
  }
  console.log('all DOM checks passed');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
