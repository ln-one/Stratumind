use super::{DotI8Fn, DotI8x4Fn, dot_i8_scalar};

#[inline]
pub(super) fn resolve_dot_i8() -> DotI8Fn {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        dot_i8_sdot_safe
    } else {
        // NEON is mandatory on AArch64.
        dot_i8_neon_safe
    }
}

#[inline]
pub(super) fn resolve_dot_i8_x4() -> Option<DotI8x4Fn> {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        Some(dot_i8_x4_sdot_safe)
    } else {
        None
    }
}

#[inline]
pub(crate) fn dot_i8_neon_safe(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    // NEON is mandatory on AArch64.
    // SAFETY: AArch64 guarantees NEON and both slices have equal length.
    unsafe { dot_i8_neon(query, vector) }
}

#[inline]
pub(crate) fn dot_i8_sdot_safe(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    // SAFETY: this wrapper is returned only after DOTPROD runtime detection.
    unsafe { dot_i8_sdot(query, vector) }
}

#[inline]
fn dot_i8_x4_sdot_safe(query: &[i8], vectors: [&[i8]; 4]) -> [i64; 4] {
    debug_assert!(vectors.iter().all(|vector| vector.len() == query.len()));
    // SAFETY: this wrapper is returned only after DOTPROD runtime detection.
    unsafe { dot_i8_x4_sdot(query, vectors) }
}

#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(query: &[i8], vector: &[i8]) -> i64 {
    use core::arch::aarch64::*;

    unsafe {
        let mut acc_0 = vdupq_n_s32(0);
        let mut acc_1 = vdupq_n_s32(0);
        let mut acc_2 = vdupq_n_s32(0);
        let mut acc_3 = vdupq_n_s32(0);
        let chunks = query.len() / 16;
        let groups = chunks / 4;

        for group in 0..groups {
            let offset = group * 64;
            let query_0 = vld1q_s8(query.as_ptr().add(offset));
            let query_1 = vld1q_s8(query.as_ptr().add(offset + 16));
            let query_2 = vld1q_s8(query.as_ptr().add(offset + 32));
            let query_3 = vld1q_s8(query.as_ptr().add(offset + 48));
            let vector_0 = vld1q_s8(vector.as_ptr().add(offset));
            let vector_1 = vld1q_s8(vector.as_ptr().add(offset + 16));
            let vector_2 = vld1q_s8(vector.as_ptr().add(offset + 32));
            let vector_3 = vld1q_s8(vector.as_ptr().add(offset + 48));

            acc_0 = vpadalq_s16(
                vpadalq_s16(acc_0, vmull_s8(vget_low_s8(query_0), vget_low_s8(vector_0))),
                vmull_high_s8(query_0, vector_0),
            );
            acc_1 = vpadalq_s16(
                vpadalq_s16(acc_1, vmull_s8(vget_low_s8(query_1), vget_low_s8(vector_1))),
                vmull_high_s8(query_1, vector_1),
            );
            acc_2 = vpadalq_s16(
                vpadalq_s16(acc_2, vmull_s8(vget_low_s8(query_2), vget_low_s8(vector_2))),
                vmull_high_s8(query_2, vector_2),
            );
            acc_3 = vpadalq_s16(
                vpadalq_s16(acc_3, vmull_s8(vget_low_s8(query_3), vget_low_s8(vector_3))),
                vmull_high_s8(query_3, vector_3),
            );
        }

        let mut offset = groups * 64;
        let mut tail_acc = vdupq_n_s32(0);
        while offset + 16 <= query.len() {
            let query_chunk = vld1q_s8(query.as_ptr().add(offset));
            let vector_chunk = vld1q_s8(vector.as_ptr().add(offset));
            tail_acc = vpadalq_s16(
                vpadalq_s16(
                    tail_acc,
                    vmull_s8(vget_low_s8(query_chunk), vget_low_s8(vector_chunk)),
                ),
                vmull_high_s8(query_chunk, vector_chunk),
            );
            offset += 16;
        }

        let acc = vaddq_s32(vaddq_s32(acc_0, acc_1), vaddq_s32(acc_2, acc_3));
        i64::from(vaddvq_s32(vaddq_s32(acc, tail_acc)))
            + dot_i8_scalar(&query[offset..], &vector[offset..])
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_sdot(query: &[i8], vector: &[i8]) -> i64 {
    use core::arch::aarch64::*;

    unsafe {
        let mut acc_0 = vdupq_n_s32(0);
        let mut acc_1 = vdupq_n_s32(0);
        let mut acc_2 = vdupq_n_s32(0);
        let mut acc_3 = vdupq_n_s32(0);
        let chunks = query.len() / 16;
        let groups = chunks / 4;

        for group in 0..groups {
            let offset = group * 64;
            let query_0 = vld1q_s8(query.as_ptr().add(offset));
            let query_1 = vld1q_s8(query.as_ptr().add(offset + 16));
            let query_2 = vld1q_s8(query.as_ptr().add(offset + 32));
            let query_3 = vld1q_s8(query.as_ptr().add(offset + 48));
            let vector_0 = vld1q_s8(vector.as_ptr().add(offset));
            let vector_1 = vld1q_s8(vector.as_ptr().add(offset + 16));
            let vector_2 = vld1q_s8(vector.as_ptr().add(offset + 32));
            let vector_3 = vld1q_s8(vector.as_ptr().add(offset + 48));
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc_0,
                a = in(vreg) query_0,
                b = in(vreg) vector_0,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc_1,
                a = in(vreg) query_1,
                b = in(vreg) vector_1,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc_2,
                a = in(vreg) query_2,
                b = in(vreg) vector_2,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc_3,
                a = in(vreg) query_3,
                b = in(vreg) vector_3,
                options(pure, nomem, nostack, preserves_flags),
            );
        }

        let mut offset = groups * 64;
        let mut tail_acc = vdupq_n_s32(0);
        while offset + 16 <= query.len() {
            let query_chunk = vld1q_s8(query.as_ptr().add(offset));
            let vector_chunk = vld1q_s8(vector.as_ptr().add(offset));
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) tail_acc,
                a = in(vreg) query_chunk,
                b = in(vreg) vector_chunk,
                options(pure, nomem, nostack, preserves_flags),
            );
            offset += 16;
        }

        let acc = vaddq_s32(vaddq_s32(acc_0, acc_1), vaddq_s32(acc_2, acc_3));
        i64::from(vaddvq_s32(vaddq_s32(acc, tail_acc)))
            + dot_i8_scalar(&query[offset..], &vector[offset..])
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i8_x4_sdot(query: &[i8], vectors: [&[i8]; 4]) -> [i64; 4] {
    use core::arch::aarch64::*;

    unsafe {
        let mut acc_0 = vdupq_n_s32(0);
        let mut acc_1 = vdupq_n_s32(0);
        let mut acc_2 = vdupq_n_s32(0);
        let mut acc_3 = vdupq_n_s32(0);
        let mut offset = 0usize;
        while offset + 16 <= query.len() {
            let query_chunk = vld1q_s8(query.as_ptr().add(offset));
            let vector_0 = vld1q_s8(vectors[0].as_ptr().add(offset));
            let vector_1 = vld1q_s8(vectors[1].as_ptr().add(offset));
            let vector_2 = vld1q_s8(vectors[2].as_ptr().add(offset));
            let vector_3 = vld1q_s8(vectors[3].as_ptr().add(offset));
            core::arch::asm!(
                "sdot {acc:v}.4s, {query:v}.16b, {vector:v}.16b",
                acc = inout(vreg) acc_0,
                query = in(vreg) query_chunk,
                vector = in(vreg) vector_0,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {query:v}.16b, {vector:v}.16b",
                acc = inout(vreg) acc_1,
                query = in(vreg) query_chunk,
                vector = in(vreg) vector_1,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {query:v}.16b, {vector:v}.16b",
                acc = inout(vreg) acc_2,
                query = in(vreg) query_chunk,
                vector = in(vreg) vector_2,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                "sdot {acc:v}.4s, {query:v}.16b, {vector:v}.16b",
                acc = inout(vreg) acc_3,
                query = in(vreg) query_chunk,
                vector = in(vreg) vector_3,
                options(pure, nomem, nostack, preserves_flags),
            );
            offset += 16;
        }

        let mut sums = [
            i64::from(vaddvq_s32(acc_0)),
            i64::from(vaddvq_s32(acc_1)),
            i64::from(vaddvq_s32(acc_2)),
            i64::from(vaddvq_s32(acc_3)),
        ];
        for coordinate in offset..query.len() {
            let query = i64::from(query[coordinate]);
            sums[0] += query * i64::from(vectors[0][coordinate]);
            sums[1] += query * i64::from(vectors[1][coordinate]);
            sums[2] += query * i64::from(vectors[2][coordinate]);
            sums[3] += query * i64::from(vectors[3][coordinate]);
        }
        sums
    }
}
