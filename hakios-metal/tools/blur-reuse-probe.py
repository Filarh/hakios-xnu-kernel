#!/usr/bin/env python3
"""blur-reuse-probe.py TRACE... : how often a frame's backdrop chain could have reused the previous frame's.

Reads metal_trace=frame captures (HakiosMetalCA.m, "The frame trace") and, for every command buffer that captures the
screen for a backdrop, builds three signatures from what was ENCODED (no pixels are read):

  src  the passes that drew the screen surface before the capture: each draw's pipeline (fs, fs_label, constant keys),
       the first 64 bytes of each buffer it uses (dbuf hex), the textures it samples (by memory; the screen itself as
       SCREEN, because the window server draws into a different surface of its pool each frame), viewport, scissor,
       geometry box and uber-shader key; and the first pass's load action, unless its first draw is an opaque quad that
       covers the target (then what it loaded is not seen);
  cap  the capture pass(es) -- the draws into the capture texture, the copy from the screen, the edge rings and the colour
       filters done in place by framebuffer fetch (Tmpc..., Tc33...) -- the same way;
  red  the first reduction that reads the capture (downsample_8, downsample_4, or the blur at full resolution).

A frame whose src and cap equal the previous frame's (for the capture of the same size) is one where reusing the
previous capture would have given the same pixels, AS FAR AS THE TRACE CAN TELL: buffers are compared on their first
64 bytes only, and the CONTENTS of the textures the screen passes sample (IOSurfaces of other processes, CA's backing
stores) are not in the trace -- so this is an upper bound for the driver-level exact reuse of 22/25's chain. A cap
that differs says WHICH draw differs (its fs_label): if it is the in-place filter (Tmpc), the material's colour filter
animates with the radius and an exact reuse of the filtered capture is impossible.

  --verbose   one line per command buffer
"""
import hashlib
import re
import sys
from collections import Counter, defaultdict

KV = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S*)')


def fields(line):
    return {k: v for k, v in KV.findall(line)}


def h(*parts):
    return hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:12]


class Draw:
    def __init__(self, f):
        self.f = f
        self.i = f.get('i')
        self.enc = f.get('enc')
        self.pipe = f.get('pipe')
        self.bufs = []      # (stage, slot, hex) of used buffers
        self.texs = []      # (stage, slot, tex id) of used textures


def parse(path):
    """yields (cb fields, encoders, draws) per command buffer record, with the file's pipe/tex/pblend tables filled"""
    pipes, texs, blends = {}, {}, {}
    cb = None
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        word = line.split(' ', 1)[0]
        if word == 'cb':
            if cb is not None:
                yield cb, pipes, texs, blends
            cb = {'f': fields(line), 'encs': {}, 'order': [], 'draws': []}
            continue
        if cb is None:
            if word in ('pipe', 'tex', 'pblend'):
                pass
            else:
                continue
        f = fields(line)
        if word == 'pipe' and 'fs' in f:
            pipes[f['id']] = f
        elif word == 'pblend':
            if f.get('att') == '0':
                blends[f['pipe']] = f.get('on') == '1'
        elif word == 'tex':
            texs[f['id']] = f
        elif word == 'enc' and cb is not None:
            e = {'i': f['i'], 'kind': f.get('kind'), 'target': None, 'load': None, 'draws': []}
            cb['encs'][f['i']] = e
            cb['order'].append(f['i'])
        elif word == 'att' and cb is not None and f.get('slot') == 'c0' and f.get('enc') in cb['encs']:
            cb['encs'][f['enc']]['target'] = f.get('tex')
            cb['encs'][f['enc']]['load'] = f.get('load')
        elif word == 'draw' and cb is not None:
            d = Draw(f)
            cb['draws'].append(d)
            if d.enc in cb['encs']:
                cb['encs'][d.enc]['draws'].append(d)
        elif word == 'dbuf' and cb is not None and 'd' in f and cb['draws'] and cb['draws'][-1].i == f['d']:
            if f.get('used') in ('1', '?'):
                cb['draws'][-1].bufs.append((f.get('stage'), f.get('slot'), f.get('hex')))
        elif word == 'dtex' and cb is not None and 'd' in f and cb['draws'] and cb['draws'][-1].i == f['d']:
            if f.get('used') in ('1', '?'):
                cb['draws'][-1].texs.append((f.get('stage'), f.get('slot'), f.get('tex')))
    if cb is not None:
        yield cb, pipes, texs, blends


def mem(texs, t):
    x = texs.get(t, {})
    return x.get('mem', 't?' + str(t))


def is_surface(texs, t):
    return texs.get(t, {}).get('surface', '0') not in ('0', '')


def label(pipes, d):
    p = pipes.get(d.pipe, {})
    name = (p.get('fs_label') or '').strip('"')
    return name if name not in ('', '-') else (p.get('fs') or '?').strip('"')


def draw_sig(pipes, texs, d, screen):
    p = pipes.get(d.pipe, {})
    sampled = []
    for stage, slot, t in d.texs:
        sampled.append((stage, slot, 'SCREEN' if screen is not None and mem(texs, t) == screen else mem(texs, t)))
    f = d.f
    return h(p.get('vs'), p.get('fs'), p.get('fs_label'), p.get('vs_key'), p.get('fs_key'), f.get('prim'), f.get('vcount'),
             f.get('vp'), f.get('sc'), f.get('geom'), f.get('spec'), f.get('inst'), sorted(d.bufs), sampled)


def covers(pipes, blends, d, enc_target, texs):
    """the draw is an opaque quad over its whole target: what the pass loaded is not seen"""
    if blends.get(d.pipe, True):
        return False
    t = texs.get(enc_target, {})
    try:
        w, hh = int(t.get('w', 0)), int(t.get('h', 0))
        x0, y0, x1, y1 = (float(v) for v in d.f.get('geom', '').split(','))
    except ValueError:
        return False
    return x0 <= 0.5 and y0 <= 0.5 and x1 >= w - 0.5 and y1 >= hh - 0.5 and d.f.get('sc', '-') in ('-', '0,0,%d,%d' % (w, hh))


REDUCE = re.compile(r'downsample_|narrow_blur_|single_pass_blur_|variable_blur')


def chains(cb, pipes, texs, blends):
    """each capture of the screen in this command buffer: (key, src, cap, red, details)"""
    encs = [cb['encs'][i] for i in cb['order'] if cb['encs'][i]['kind'] in ('render', 'parallel')]
    out = []
    for k, e in enumerate(encs):
        tgt = e['target']
        if tgt is None or is_surface(texs, tgt):
            continue
        # a capture: this pass samples a surface that an earlier pass of this buffer drew (the screen)
        screens = set()
        for d in e['draws']:
            for _, _, t in d.texs:
                if is_surface(texs, t) and any(x['target'] is not None and mem(texs, x['target']) == mem(texs, t) for x in encs[:k]):
                    screens.add(mem(texs, t))
        if not screens:
            continue
        screen = sorted(screens)[0]
        cap_mem = mem(texs, tgt)
        before = [x for x in encs[:k] if x['target'] is not None and mem(texs, x['target']) == screen]
        src_parts, proof = [], None
        for n, x in enumerate(before):
            if n == 0:
                first = x['draws'][0] if x['draws'] else None
                proof = x['load'] in ('clear', 'dontcare') or (first is not None and covers(pipes, blends, first, x['target'], texs))
                src_parts.append('load' if not proof else 'noload')
            src_parts.extend(draw_sig(pipes, texs, d, screen) for d in x['draws'])
        # the capture's own passes: this one and the next ones into the same texture (in-place filters, edges)
        cap_draws = []
        j = k
        while j < len(encs) and encs[j]['target'] is not None and mem(texs, encs[j]['target']) == cap_mem:
            cap_draws.extend(encs[j]['draws'])
            j += 1
        red = None
        for x in encs[j:]:
            for d in x['draws']:
                if REDUCE.search((pipes.get(d.pipe, {}).get('fs') or '')) and any(mem(texs, t) == cap_mem for _, _, t in d.texs):
                    red = d
                    break
            if red is not None:
                break
        key = (texs.get(tgt, {}).get('w'), texs.get(tgt, {}).get('h'), len(out))
        out.append({
            'key': key,
            'src': h(*src_parts), 'src_n': sum(len(x['draws']) for x in before), 'proof': bool(proof),
            'cap': h(*(draw_sig(pipes, texs, d, screen) for d in cap_draws)),
            'cap_each': [(label(pipes, d), draw_sig(pipes, texs, d, screen)) for d in cap_draws],
            'red': draw_sig(pipes, texs, red, screen) if red is not None else None,
            'red_name': (pipes.get(red.pipe, {}).get('fs') or '').strip('"') if red is not None else '-',
            'chain': [label(pipes, d) for x in encs[j:j + 6] for d in x['draws']][:8],
        })
    return out


def main(argv):
    verbose = '--verbose' in argv
    paths = [a for a in argv if not a.startswith('--')]
    if not paths:
        print(__doc__)
        return 2
    total = Counter()
    differs = Counter()
    by_red = defaultdict(Counter)
    for path in paths:
        prev = {}
        for cb, pipes, texs, blends in parse(path):
            for c in chains(cb, pipes, texs, blends):
                total['captures'] += 1
                p = prev.get(c['key'])
                prev[c['key']] = c
                if p is None:
                    total['first'] += 1
                    continue
                total['compared'] += 1
                same_src = c['src'] == p['src'] and c['proof'] and p['proof']
                same_cap = c['cap'] == p['cap']
                same_red = c['red'] is not None and c['red'] == p['red']
                total['src_same'] += same_src
                total['src_same_but_load'] += c['src'] == p['src'] and not (c['proof'] and p['proof'])
                total['cap_same'] += same_cap
                total['reuse_capture'] += same_src and same_cap
                total['reuse_reduction'] += same_src and same_cap and same_red
                by_red[c['red_name']]['frames'] += 1
                by_red[c['red_name']]['reuse_capture'] += same_src and same_cap
                if not same_cap:
                    old = dict(p['cap_each'])
                    for name, sig in c['cap_each']:
                        if old.get(name) != sig:
                            differs[name] += 1
                if verbose:
                    print('%s cb %s capture %sx%s: src %s (%d draws%s) cap %s red %s %s | %s' % (
                        path.rsplit('/', 1)[-1], cb['f'].get('seq'), c['key'][0], c['key'][1],
                        'same' if same_src else 'diff', c['src_n'], '' if c['proof'] else ', loads', 'same' if same_cap else 'diff',
                        c['red_name'], 'same' if same_red else 'diff', ' '.join(c['chain'])))
    print('captures of the screen: %d (first of their size: %d, compared with the previous frame: %d)' % (
        total['captures'], total['first'], total['compared']))
    if total['compared']:
        n = total['compared']
        print('  screen passes before the capture identical (and not depending on what they loaded): %d (%.0f%%)' % (total['src_same'], 100.0 * total['src_same'] / n))
        print('  ... identical but the first pass loads the old surface (not provable): %d' % total['src_same_but_load'])
        print('  capture passes identical (copy, edges, in-place filters): %d (%.0f%%)' % (total['cap_same'], 100.0 * total['cap_same'] / n))
        print('  => capture reusable (upper bound): %d (%.0f%%); reduction reusable too: %d' % (
            total['reuse_capture'], 100.0 * total['reuse_capture'] / n, total['reuse_reduction']))
        for name, c in sorted(by_red.items(), key=lambda kv: -kv[1]['frames']):
            print('    first reduction %-28s frames %4d, capture reusable %4d' % (name, c['frames'], c['reuse_capture']))
        if differs:
            print('  draws of the capture that changed from one frame to the next (fs_label: frames):')
            for name, k in differs.most_common(12):
                print('    %-40s %d' % (name, k))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
