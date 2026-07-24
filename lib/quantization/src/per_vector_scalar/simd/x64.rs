use super::{DotI8Fn, dot_i8_scalar};

#[inline]
pub(super) fn resolve_dot_i8() -> DotI8Fn {
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vnni")
    {
        return dot_i8_avx512_vnni_safe;
    }
    if std::is_x86_feature_detected!("avx2") {
        return dot_i8_avx2_safe;
    }
    if std::is_x86_feature_detected!("sse4.1") {
        return dot_i8_sse41_safe;
    }
    dot_i8_scalar
}

#[inline]
pub(crate) fn dot_i8_sse41_safe(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    // SAFETY: this wrapper is returned only after SSE4.1 runtime detection.
    unsafe { dot_i8_sse41(query, vector) }
}

#[inline]
pub(crate) fn dot_i8_avx2_safe(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    // SAFETY: this wrapper is returned only after AVX2 runtime detection.
    unsafe { dot_i8_avx2(query, vector) }
}

#[inline]
pub(crate) fn dot_i8_avx512_vnni_safe(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    // SAFETY: this wrapper is returned only after AVX512 runtime detection.
    unsafe { dot_i8_avx512_vnni(query, vector) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn dot_i8_sse41(query: &[i8], vector: &[i8]) -> i64 {
    use core::arch::x86_64::*;

    unsafe {
        let mut acc = _mm_setzero_si128();
        let mut offset = 0;
        while offset + 16 <= query.len() {
            let query_bytes = _mm_loadu_si128(query.as_ptr().add(offset).cast());
            let vector_bytes = _mm_loadu_si128(vector.as_ptr().add(offset).cast());
            let query_low = _mm_cvtepi8_epi16(query_bytes);
            let query_high = _mm_cvtepi8_epi16(_mm_srli_si128(query_bytes, 8));
            let vector_low = _mm_cvtepi8_epi16(vector_bytes);
            let vector_high = _mm_cvtepi8_epi16(_mm_srli_si128(vector_bytes, 8));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(query_low, vector_low));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(query_high, vector_high));
            offset += 16;
        }
        let mut lanes = [0i32; 4];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), acc);
        lanes.into_iter().map(i64::from).sum::<i64>()
            + dot_i8_scalar(&query[offset..], &vector[offset..])
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(query: &[i8], vector: &[i8]) -> i64 {
    use core::arch::x86_64::*;

    unsafe {
        let mut acc = _mm256_setzero_si256();
        let mut offset = 0;
        while offset + 32 <= query.len() {
            let query_bytes = _mm256_loadu_si256(query.as_ptr().add(offset).cast());
            let vector_bytes = _mm256_loadu_si256(vector.as_ptr().add(offset).cast());
            let query_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(query_bytes));
            let query_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(query_bytes, 1));
            let vector_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vector_bytes));
            let vector_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vector_bytes, 1));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(query_low, vector_low));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(query_high, vector_high));
            offset += 32;
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
        lanes.into_iter().map(i64::from).sum::<i64>()
            + dot_i8_scalar(&query[offset..], &vector[offset..])
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_i8_avx512_vnni(query: &[i8], vector: &[i8]) -> i64 {
    use core::arch::x86_64::*;

    unsafe {
        let mut acc = _mm512_setzero_si512();
        let mut offset = 0;
        while offset + 64 <= query.len() {
            let query_bytes = _mm512_loadu_si512(query.as_ptr().add(offset).cast());
            let vector_bytes = _mm512_loadu_si512(vector.as_ptr().add(offset).cast());
            let query_low = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(query_bytes));
            let query_high = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(query_bytes, 1));
            let vector_low = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vector_bytes));
            let vector_high = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vector_bytes, 1));
            acc = _mm512_dpwssd_epi32(acc, query_low, vector_low);
            acc = _mm512_dpwssd_epi32(acc, query_high, vector_high);
            offset += 64;
        }
        i64::from(_mm512_reduce_add_epi32(acc)) + dot_i8_scalar(&query[offset..], &vector[offset..])
    }
}
