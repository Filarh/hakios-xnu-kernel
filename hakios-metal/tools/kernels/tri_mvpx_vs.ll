; tri_mvpx_vs, the AIR side: a vertex function whose position is a matrix times a [[stage_in]] attribute, written as the
; LLVM IR a Metal compiler would emit for
;
;     struct In  { float4 position [[attribute(0)]]; float4 color [[attribute(1)]]; };
;     struct Out { float4 position [[position]]; float4 color; };
;     vertex Out tri_mvpx_vs(In in [[stage_in]], constant float4x4 &m [[buffer(1)]])
;     {
;       return Out{ m * in.position, in.color };   // compiled with -fno-fast-math
;     }
;
; Hand-written, ours, MIT: only the SHAPE of the metadata is Apple's convention (tri_attr_vs.ll for the attributes,
; tri_cflat_fs.ll for the constant buffer).
;
; Why it exists: tri_mvp_vs's twin without fast-math flags. The products and the sums are then each an operation of its
; own for ir3 -- mul.f, add.f and the movs that put the position in its registers -- where tri_mvp_vs's contract into
; mad.f32: the other instructions CommandBuffer::setCull's interpreter runs (cull.hpp), on the same pictures.
;
;   llvm-as tri_mvpx_vs.ll -o tri_mvpx_vs.bc
;   ir3_compiler --chip-id 0x06030500 tri_mvpx_vs.bc

source_filename = "tri_mvpx_vs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

define <{ <4 x float>, <4 x float> }> @tri_mvpx_vs(<4 x float> %pos, <4 x float> %color, ptr addrspace(2) %m) local_unnamed_addr {
  %c0 = load <4 x float>, ptr addrspace(2) %m, align 16
  %p1 = getelementptr inbounds <4 x float>, ptr addrspace(2) %m, i64 1
  %c1 = load <4 x float>, ptr addrspace(2) %p1, align 16
  %p2 = getelementptr inbounds <4 x float>, ptr addrspace(2) %m, i64 2
  %c2 = load <4 x float>, ptr addrspace(2) %p2, align 16
  %p3 = getelementptr inbounds <4 x float>, ptr addrspace(2) %m, i64 3
  %c3 = load <4 x float>, ptr addrspace(2) %p3, align 16
  %x = shufflevector <4 x float> %pos, <4 x float> undef, <4 x i32> zeroinitializer
  %y = shufflevector <4 x float> %pos, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %z = shufflevector <4 x float> %pos, <4 x float> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %w = shufflevector <4 x float> %pos, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %t0 = fmul <4 x float> %c0, %x
  %t1 = fmul <4 x float> %c1, %y
  %s1 = fadd <4 x float> %t1, %t0
  %t2 = fmul <4 x float> %c2, %z
  %s2 = fadd <4 x float> %t2, %s1
  %t3 = fmul <4 x float> %c3, %w
  %s3 = fadd <4 x float> %t3, %s2
  %r0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %s3, 0
  %r1 = insertvalue <{ <4 x float>, <4 x float> }> %r0, <4 x float> %color, 1
  ret <{ <4 x float>, <4 x float> }> %r1
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.vertex = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_mvpx_vs, !3, !6}
!3 = !{!4, !5}
!4 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{!"air.vertex_output", !"generated(5colorDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!6 = !{!7, !8, !9}
!7 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!8 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!9 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 64, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4x4", !"air.arg_name", !"m"}
