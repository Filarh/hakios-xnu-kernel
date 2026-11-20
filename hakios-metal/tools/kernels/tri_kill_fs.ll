; tri_kill_fs, a fragment function that discards: tri_cflat_fs with a checkerboard of 4 x 4 pixels thrown away, the LLVM IR
; a Metal compiler would emit for
;
;     fragment float4 tri_kill_fs(float4 pos [[position]], constant float4 &c [[buffer(0)]])
;     {
;       if (((uint(pos.x) ^ uint(pos.y)) & 4) != 0) discard_fragment();
;       return c;
;     }
;
; Hand-written, ours, MIT: nothing in it comes from Apple; only the SHAPE of the metadata is Apple's convention (tri_ptex_fs.ll
; for the position, tri_cflat_fs.ll for the buffer).
;
; Why it exists: hakios-metal-hsrcheck draws it behind and in front of an opaque draw, with hidden surface removal by
; depth and without (CommandBuffer::setHsr). A draw that discards is not opaque -- what is under the pixels it throws
; away stays -- so it is tested against the prepass's depth in its place, early, since it writes no depth: the
; picture must be the same with HSR as without, pixel for pixel, the checkerboard and all.
;
;   llvm-as tri_kill_fs.ll -o tri_kill_fs.bc
;   python3 -I tools/bc-header.py tri_kill_fs.bc tri_kill_fs > tests/fixtures/tri_kill_fs_bc.hpp

source_filename = "tri_kill_fs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

declare void @air.discard_fragment()

define <4 x float> @tri_kill_fs(<4 x float> %pos, ptr addrspace(2) %c) local_unnamed_addr {
entry:
  %x = extractelement <4 x float> %pos, i32 0
  %y = extractelement <4 x float> %pos, i32 1
  %xi = fptoui float %x to i32
  %yi = fptoui float %y to i32
  %m = xor i32 %xi, %yi
  %b = and i32 %m, 4
  %k = icmp ne i32 %b, 0
  br i1 %k, label %kill, label %keep

kill:
  tail call void @air.discard_fragment()
  br label %keep

keep:
  %v = load <4 x float>, ptr addrspace(2) %c, align 16
  ret <4 x float> %v
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.fragment = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_kill_fs, !3, !5}
!3 = !{!4}
!4 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!5 = !{!6, !7}
!6 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!7 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
