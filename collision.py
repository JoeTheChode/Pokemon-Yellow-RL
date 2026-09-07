"""Render ROM collision for Route 13/14/15 and overlay the live swarm heatmap.

walkable(y,x) = blockset[blk[(y//2)*wb + (x//2)] * 16 + (2*(y%2)+1)*4 + 2*(x%2)]
                in that tileset's coll_tiles  (bottom-left tile of the 2x2 quadrant)
"""
import json
import re
import collections
import pathlib

SRC = pathlib.Path('/tmp/pokeyellow-master')
HEAT = {
    'Route13': (294, 326, 60, 18),
    'Route14': (274, 326, 20, 54),
    'Route15': (214, 362, 60, 18),
}


def load_coll_tiles():
    text = (SRC / 'data/tilesets/collision_tile_ids.asm').read_text()
    out = {}
    pending = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^(\w+)_Coll::', line)
        if m:
            pending.append(m.group(1).upper())
            continue
        if pending and line.startswith('coll_tiles'):
            vals = [
                int(tok.strip()[1:], 16)
                for tok in line[len('coll_tiles'):].split(';')[0].split(',')
                if tok.strip().startswith('$')
            ]
            for name in pending:
                out[name] = vals
            pending = []
    return out


def map_tileset(name):
    text = (SRC / f'data/maps/headers/{name}.asm').read_text()
    m = re.search(r'map_header\s+\w+,\s*\w+,\s*(\w+)', text)
    return m.group(1)


def map_size(name):
    """(width_blocks, height_blocks) from constants/map_constants.asm."""
    const = re.sub(r'(?<=[a-z])(?=\d)', '_', name).upper()  # Route13 -> ROUTE_13
    text = (SRC / 'constants/map_constants.asm').read_text()
    m = re.search(rf'map_const\s+{const},\s*(\d+),\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


coll = load_coll_tiles()
print('coll tilesets:', {k: len(v) for k, v in coll.items()})

heat_total = collections.Counter()
hm = json.load(open('viewer_heatmap.json'))
for env, rows in hm['by_env'].items():
    for gx, gy, count, ts in rows:
        heat_total[(gx, gy)] += count

for name in ('Route13', 'Route14', 'Route15'):
    tileset = map_tileset(name)
    size = map_size(name)
    blk = (SRC / f'maps/{name}.blk').read_bytes()
    wb, hb = size
    print(f'\n===== {name}: tileset={tileset} blocks={wb}x{hb} '
          f'tiles={wb*2}x{hb*2} blk={len(blk)} =====')
    bst = (SRC / f'gfx/blocksets/{tileset.lower()}.bst').read_bytes()
    tiles = coll.get(tileset.upper(), [])
    if not tiles:
        print('  !! no coll tiles for', tileset)
        continue
    allowed = set(tiles)
    W, H = wb * 2, hb * 2
    x0, y0, hw, hh = HEAT[name]
    walk = {}
    for y in range(H):
        for x in range(W):
            bi = blk[(y // 2) * wb + (x // 2)]
            ti = bi * 16 + (2 * (y % 2) + 1) * 4 + 2 * (x % 2)
            walk[(y, x)] = bst[ti] in allowed if ti < len(bst) else False
    # overlay
    print('     ' + ''.join(str(x // 10 % 10) for x in range(W)))
    print('     ' + ''.join(str(x % 10) for x in range(W)))
    agree = dis = 0
    for y in range(H):
        row = ''
        for x in range(W):
            visited = heat_total.get((x0 + x, y0 + y), 0) > 0
            w = walk[(y, x)]
            if visited and w:
                row += 'O'
                agree += 1
            elif visited and not w:
                row += '!'
                dis += 1
            elif w:
                row += '.'
            else:
                row += '#'
        print(f'{y:3d}  {row}')
    print(f'  heatmap-visited & ROM-walkable: {agree}, '
          f'visited but ROM says BLOCKED: {dis}')
