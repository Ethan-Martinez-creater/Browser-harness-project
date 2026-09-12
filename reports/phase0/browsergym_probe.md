# BrowserGym Environment Probe — Phase 0 Task 1

Date: 2026-09-12

## Environment

| Item | Value |
|---|---|
| Python | 3.11.16 (uv-managed, project `.venv`) |
| browsergym-core | 0.14.3 |
| browsergym-miniwob | 0.14.3 |
| gymnasium | 1.3.0 |
| playwright | 1.44.0 |
| Chromium | 125.0.6422.26 (playwright build v1117, headless) |
| openai | 3.13.0 |
| pydantic | 2.13.5 |
| platform | Windows-10-10.0.19045-SP0 |

## Probe setup

- Task: `browsergym/miniwob.click-test`
- MiniWoB HTML source: local `miniwob-plusplus` repository cloned to
  `third_party/miniwob-plusplus`, served through a `file://` URL via the
  `MINIWOB_URL` environment variable. No external network is needed at run time.
- Script: `scripts/browsergym_probe.py` (repeatable).

## Findings

1. `env.reset(seed=0)` works; observation keys are:
   `active_page_index, axtree_object, chat_messages, dom_object, elapsed_time,
   extra_element_properties, focused_element_bid, goal, goal_object,
   last_action, last_action_error, open_pages_titles, open_pages_urls,
   screenshot, url`.
2. In browsergym 0.14.3 the raw observation exposes **objects**, not text:
   there is no `axtree_txt`/`dom_txt` key. Text must be produced with
   `browsergym.utils.obs.flatten_axtree_to_str(obs["axtree_object"])`. Our
   `ObservationNormalizer` handles this.
3. Right after `reset()` the axtree can be empty (first frame not rendered).
   A `noop()` step refreshes the observation; the normalizer/adapter accounts
   for this.
4. Axtree line format is `[bid] role 'name'`, e.g. `[13] button 'Click Me!'`.
5. High-level action `click(bid='13')` executed without error.
6. Result: `reward=1.0`, `terminated=True` → task ended correctly.
7. `env.close()` works; environment must be closed in `finally`.

## Probe result

```text
[probe] reset: OK
[probe] goal: Click the button.
[probe] axtree head: RootWebArea 'Click Test Task', focused |  [13] button 'Click Me!'
[probe] found bid: 13
[probe] action: click(bid='13')
[probe] last_action_error: none
[probe] reward: 1.0
[probe] terminated: True
[probe] PROBE RESULT: SUCCESS
```

## Consequences for the harness

- BrowserGym types stay inside `env/browsergym_adapter.py` only.
- `MINIWOB_URL` must be resolvable; the adapter falls back to the local
  `third_party/miniwob-plusplus/miniwob/html/miniwob/` checkout when present.
- Observation normalization (`flatten_axtree_to_str`, deterministic
  truncation) is a dedicated module.
