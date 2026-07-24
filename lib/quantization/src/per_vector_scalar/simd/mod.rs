//! Exact signed-i8 dot products for Per-Vector Scalar Quantization.

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x64;

/// Largest dimension whose worst-case `127 * 127` dot product fits in i32.
pub(super) const MAX_I32_DIMENSION: usize = i32::MAX as usize / (127 * 127);
pub(super) type DotI8Fn = fn(&[i8], &[i8]) -> i64;
pub(super) type DotI8x4Fn = fn(&[i8], [&[i8]; 4]) -> [i64; 4];

#[inline]
pub(super) fn dot_i8(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    resolve_dot_i8(query.len())(query, vector)
}

#[inline]
pub(super) fn resolve_dot_i8(dimension: usize) -> DotI8Fn {
    if dimension > MAX_I32_DIMENSION {
        return dot_i8_scalar;
    }

    #[cfg(target_arch = "aarch64")]
    {
        return arm::resolve_dot_i8();
    }

    #[cfg(target_arch = "x86_64")]
    {
        return x64::resolve_dot_i8();
    }

    #[allow(unreachable_code)]
    dot_i8_scalar
}

#[inline]
pub(super) fn resolve_dot_i8_x4(dimension: usize) -> Option<DotI8x4Fn> {
    if dimension > MAX_I32_DIMENSION {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    {
        return arm::resolve_dot_i8_x4();
    }

    #[allow(unreachable_code)]
    None
}

#[inline]
pub(super) fn selected_kernel_name(dimension: usize, contiguous: bool) -> &'static str {
    if dimension > MAX_I32_DIMENSION {
        return "scalar-i64";
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return if contiguous {
                "aarch64-sdot-x4"
            } else {
                "aarch64-sdot"
            };
        }
        return "aarch64-neon";
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
        {
            return "x86-avx512-vnni";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "x86-avx2";
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return "x86-sse4.1";
        }
    }

    #[allow(unreachable_code)]
    "scalar-i32"
}

#[inline]
pub(super) fn dot_i8_scalar(query: &[i8], vector: &[i8]) -> i64 {
    query
        .iter()
        .zip(vector)
        .map(|(&query, &vector)| i64::from(query) * i64::from(vector))
        .sum()
}

#[inline]
#[cfg(test)]
pub(super) fn dot_i8_portable_i32(query: &[i8], vector: &[i8]) -> i64 {
    debug_assert_eq!(query.len(), vector.len());
    if query.len() <= MAX_I32_DIMENSION {
        i64::from(
            query
                .iter()
                .zip(vector)
                .map(|(&query, &vector)| i32::from(query) * i32::from(vector))
                .sum::<i32>(),
        )
    } else {
        dot_i8_scalar(query, vector)
    }
}

#[inline]
#[cfg(test)]
fn dot_i8_x4_scalar(query: &[i8], vectors: [&[i8]; 4]) -> [i64; 4] {
    [
        dot_i8_portable_i32(query, vectors[0]),
        dot_i8_portable_i32(query, vectors[1]),
        dot_i8_portable_i32(query, vectors[2]),
        dot_i8_portable_i32(query, vectors[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_matches_scalar_for_padded_and_tail_dimensions() {
        for dimension in [
            1, 15, 16, 17, 31, 32, 33, 128, 192, 256, 384, 768, 1024, 1536,
        ] {
            let query: Vec<_> = (0..dimension)
                .map(|i| (((i * 37 + dimension * 11) % 255) as i16 - 127) as i8)
                .collect();
            let vector: Vec<_> = (0..dimension)
                .map(|i| (((i * 53 + dimension * 7) % 255) as i16 - 127) as i8)
                .collect();
            assert_eq!(
                dot_i8(&query, &vector),
                dot_i8_scalar(&query, &vector),
                "dimension={dimension}",
            );
            assert_eq!(
                dot_i8_portable_i32(&query, &vector),
                dot_i8_scalar(&query, &vector),
                "portable i32 dimension={dimension}",
            );

            let vectors = [
                vector.clone(),
                vector.iter().copied().rev().collect(),
                vector.iter().map(|value| value.saturating_neg()).collect(),
                vec![0; dimension],
            ];
            let slices = [
                vectors[0].as_slice(),
                vectors[1].as_slice(),
                vectors[2].as_slice(),
                vectors[3].as_slice(),
            ];
            let expected = slices.map(|candidate| dot_i8_scalar(&query, candidate));
            assert_eq!(
                resolve_dot_i8_x4(dimension).unwrap_or(dot_i8_x4_scalar)(&query, slices),
                expected,
                "batch4 dimension={dimension}",
            );
        }
    }

    #[test]
    fn extrema_do_not_overflow_supported_dimension() {
        let dimension = 4096;
        let positive = vec![127; dimension];
        let negative = vec![-127; dimension];
        assert_eq!(
            dot_i8(&positive, &negative),
            -(dimension as i64) * 127 * 127,
        );
    }
}
