# Dirty Run 07 After Brock Waypoints

Source: `manual_map_07_after_brock_clean.log` / `manual_07_after_brock_clean.jsonl`

This run reached well past the filename's "after Brock" label: Cascade Badge obtained,
Charmander and Bulbasaur picked up, Nugget Bridge completed, Bill/S.S. Anne/Cut reached,
and Lt. Surge started. The RAM badge and battle/text fields were not reliable, so this
file is navigation reference only, not clean demonstration data.

## Training Waypoints Used For 07_beat_brock

These are the coordinates currently encoded in `train.py` for teaching
Route 3 -> Mt. Moon -> Cerulean traversal.

```text
PewterCity map=2:
  (17,37)

Route3 map=14:
  (10,0) -> (9,8) -> (9,11) -> (6,11) -> (6,14) -> (4,19)
  -> (6,21) -> (6,27) -> (8,33) -> (5,47) -> (11,56)
  -> (8,59) -> (0,59)

Route4 west / Mt. Moon entrance map=15:
  (17,9) -> (13,12) -> (6,12) -> (6,18)

MtMoonPC map=68:
  (7,3) -> (3,3) -> (5,9) -> (6,3)

MtMoon1F map=59:
  (35,14) -> (27,14) -> (22,20) -> (12,31) -> (7,30)
  -> (7,15) -> (12,16) -> (18,10) -> (5,10) -> (5,6)

MtMoonB1F map=60:
  (5,5) -> (17,5) -> (17,20)

MtMoonB2F map=61:
  (17,21) -> (14,27) -> (16,36) -> (22,34) -> (31,34)
  -> (31,10) -> (18,13) -> (8,13) -> (4,3) -> (7,4)

MtMoonB1F exit path map=60:
  (7,5) -> (3,23) -> (3,27)

Route4 east / Cerulean approach map=15:
  (5,24) -> (6,41) -> (10,61) -> (6,78) -> (11,82)

CeruleanCity map=3:
  (35,25) -> (31,36) -> (20,29) -> (18,13)
```

## Dirty Later-Run Landmarks

The same dirty run continued beyond Cerulean and is useful only as a rough scout:

```text
CeruleanCity map=3:
  entered from Route4 at about (35,25)
  visited center/buildings around (18,13), (15,13), and surrounding city paths

Nugget Bridge / Bill / post-Cerulean maps:
  later map ids in the log include 5, 16, 17, 22, 63, 71, 74, 89, 92,
  94, 95, 96, 97, 100, 101, 102, and 119.
  These need clean labels and validation before they are used for curriculum shaping.
```
