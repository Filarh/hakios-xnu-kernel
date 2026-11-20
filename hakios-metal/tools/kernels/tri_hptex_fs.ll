; tri_hptex_fs: a texture sampled at the fragment's own pixel, the LLVM IR a Metal compiler would emit for
;
;     fragment half4 tri_hptex_fs(float4 pos [[position]], texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]])
;     {
;       return t.sample(s, pos.xy / float2(t.get_width(), t.get_height()));   // [[color(0)]]
;     }
;
; Drawn over a target of the texture's size (any vertex function covering it: tri_color_vs), pixel (x, y) samples
; texel (x, y) at its centre: what hakios-metal-ycbcrcheck compares with the processor's conversion pixel by pixel.
; tri_tex_fs samples at tri_color_vs's colour, the barycentric weights of each triangle's first two vertices, which
; is a picture of the texture but not texel (x, y) at pixel (x, y). The size queries are txs, which a YCbCr slot's
; conversion leaves alone.
;
; Hand-written, ours, MIT: only the SHAPE of the metadata and of the intrinsic signatures is Apple's convention
; (tri_shadow_fs.ll for the position, tex_size_h.ll for the size).

source_filename = "tri_hptex_fs"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-ios16.5.0"

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)

define <4 x half> @tri_hptex_fs(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %s) local_unnamed_addr {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 0)
  %wf = uitofp i32 %w to float
  %hf = uitofp i32 %h to float
  %s0 = insertelement <2 x float> poison, float %wf, i32 0
  %size = insertelement <2 x float> %s0, float %hf, i32 1
  %uv = fdiv <2 x float> %xy, %size
  %r = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %r, 0
  ret <4 x half> %v
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.fragment = !{!2}

!0 = !{i32 2, i32 5, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @tri_hptex_fs, !3, !5}
!3 = !{!4}
!4 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!5 = !{!6, !7, !8}
!6 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!7 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"t"}
!8 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
