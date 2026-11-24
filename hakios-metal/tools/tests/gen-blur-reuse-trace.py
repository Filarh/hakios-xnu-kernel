#!/usr/bin/env python3
"""writes a small metal_trace=frame capture with 3 command buffers of a Control Center-like backdrop chain:
cb 1 and 2 encode the same work (into different screen surfaces of the pool), cb 3 changes the in-place filter's
constants (Tmpc) -- the case of a material whose colour filter animates with the radius."""
import sys
out = []
p = out.append
p('pipe id=1 label="main" vs="fixed_vert_lph_spc" fs="fixed_frag_lph_cph" vs_label="x" fs_label="TimgA2Xhf_Isrc" vs_key=1 fs_key=2')
p('pblend pipe=1 att=0 fmt=bgra8 on=0 rgb=add alpha=add src=one dst=zero srca=one dsta=zero mask=rgba')
p('pipe id=2 label="tmpc" vs="fixed_vert_lph_spc" fs="fixed_frag_lph_cph" vs_label="x" fs_label="Tmpc49_Idst" vs_key=1 fs_key=3')
p('pipe id=3 label="ds8" vs="downsample_vert" fs="downsample_8_frag_lph" vs_label="-" fs_label="-" vs_key=1 fs_key=4')
p('tex id=10 mem=t10 w=752 h=1626 surface=0 label="capture"')
p('tex id=11 mem=t11 w=99 h=209 surface=0 label="reduced"')
p('tex id=12 mem=t12 w=750 h=1624 surface=0 label="wallpaper"')
for seq, (surf, tmpc) in enumerate([(31, 'aabb'), (32, 'aabb'), (33, 'ccdd')], 1):
    p('cb seq=%d capture=1 serial=%d label="CA" status=completed' % (seq, seq))
    p('tex id=%d mem=s%d w=750 h=1624 surface=%d label="screen"' % (20 + seq, surf, surf))
    p('enc i=0 kind=render t=0 rt=750x1624')
    p('att enc=0 slot=c0 tex=%d lvl=0 slice=0 plane=0 load=load store=store clear=0,0,0,0' % (20 + seq))
    p('draw i=0 enc=0 pipe=1 prog="p" prim=tri vstart=0 vcount=6 inst=1 binst=0 vp=- sc=- target=750x1624 geom=0.0,0.0,750.0,1624.0 spec=hit:1 core=0 n=1 res=ok')
    p('dbuf d=0 stage=vs slot=0 buf=0x%x len=4096 off=%d used=1 hex=00112233' % (0x1000 * seq, 64 * seq))
    p('dtex d=0 stage=fs slot=0 tex=12 used=1')
    p('enc i=1 kind=render t=0 rt=752x1626')
    p('att enc=1 slot=c0 tex=10 lvl=0 slice=0 plane=0 load=dontcare store=store clear=0,0,0,0')
    p('draw i=1 enc=1 pipe=1 prog="p" prim=tri vstart=0 vcount=6 inst=1 binst=0 vp=- sc=- target=752x1626 geom=1.0,1.0,751.0,1625.0 spec=hit:1 core=1 n=1 res=ok')
    p('dtex d=1 stage=fs slot=0 tex=%d used=1' % (20 + seq))
    p('draw i=2 enc=1 pipe=2 prog="p" prim=tri vstart=0 vcount=6 inst=1 binst=0 vp=- sc=- target=752x1626 geom=0.0,0.0,752.0,1626.0 spec=hit:2 core=1 n=1 res=ok')
    p('dbuf d=2 stage=fs slot=1 buf=0x9000 len=256 off=0 used=1 hex=%s' % tmpc)
    p('enc i=2 kind=render t=0 rt=99x209')
    p('att enc=2 slot=c0 tex=11 lvl=0 slice=0 plane=0 load=clear store=store clear=0,0,0,0')
    p('draw i=3 enc=2 pipe=3 prog="p" prim=tri vstart=0 vcount=6 inst=1 binst=0 vp=- sc=- target=99x209 geom=0.0,0.0,99.0,209.0 spec=none core=2 n=1 res=ok')
    p('dbuf d=3 stage=fs slot=0 buf=0x7000 len=256 off=0 used=1 hex=feed')
    p('dtex d=3 stage=fs slot=0 tex=10 used=1')
    p('endcb seq=%d' % seq)
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
