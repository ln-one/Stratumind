use super::*;

pub(super) fn quantization_bounds<D>(
    point_id: PointOffsetType,
    query: &EncodedQueryPerVectorScalar,
    document: PerVectorScalarRowStats,
    document_codes: &[i8],
    dot: D,
) -> Result<PerVectorScalarBounds, PerVectorScalarError>
where
    D: Fn(&[i8], &[i8]) -> i64,
{
    let integer_dot = dot(&query.codes, document_codes);
    quantization_bounds_from_integer_dot(point_id, query, document, integer_dot)
}

pub(super) fn quantization_bounds_from_integer_dot(
    point_id: PointOffsetType,
    query: &EncodedQueryPerVectorScalar,
    document: PerVectorScalarRowStats,
    integer_dot: i64,
) -> Result<PerVectorScalarBounds, PerVectorScalarError> {
    let integer = integer_dot as f64;
    let query_scale = f64::from(query.scale);
    let document_scale = f64::from(document.scale);
    let center = integer * query_scale * document_scale;

    let document_original = f64::from(document.original_norm_upper);
    let document_residual = f64::from(document.residual_norm_upper);
    // One application of the triangle inequality is sufficient:
    //
    // q·v - q̂·v̂ = (q-q̂)·v + q̂·(v-v̂).
    //
    // The previous implementation also evaluated the symmetric decomposition
    // q·(v-v̂) + (q-q̂)·v̂ and kept the smaller result. Real-data gates showed
    // only a few ten-thousandths less f32 refinement, while every document paid
    // for the second set of directed multiplications. Keep the single strict
    // bound used by the Compact reference.
    let error = query.residual_norm_upper * document_original
        + query.reconstructed_norm_upper * document_residual;
    let magnitude = center.abs() + error;
    let arithmetic_guard = (magnitude * CERTIFICATE_F64_GUARD).next_up();
    let radius = (error + arithmetic_guard).next_up();
    let quantization_lower = (center - radius).next_down();
    let quantization_upper = (center + radius).next_up();

    if [center, quantization_lower, quantization_upper]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return Err(PerVectorScalarError::NonFinite);
    }
    Ok(PerVectorScalarBounds {
        point_id,
        center,
        document_original_norm_upper: document_original,
        quantization_lower,
        quantization_upper,
    })
}

#[inline(always)]
pub(super) fn final_bounds_from_integer_dot(
    query: &EncodedQueryPerVectorScalar,
    document: PerVectorScalarRowStats,
    integer_dot: i64,
    final_guard_factor: f64,
) -> (f64, f64) {
    let center = integer_dot as f64 * f64::from(query.scale) * f64::from(document.scale);
    let document_original = f64::from(document.original_norm_upper);
    let document_residual = f64::from(document.residual_norm_upper);
    let quantization_error = query.residual_norm_upper * document_original
        + query.reconstructed_norm_upper * document_residual;
    // Reuse the production Compact certificate's single floating envelope.
    // This covers both f64 construction arithmetic and the authoritative f32
    // scorer without layering a second pair of directed endpoints.
    let guard_scale =
        center.abs() + quantization_error + query.original_norm_upper * document_original;
    // Query encoding, immutable-row validation and the Segment-level guard
    // check prove these values finite before the hot loop begins. Advance the
    // rounded guard and endpoints by one representable value without paying
    // `f64::next_up/down`'s general NaN/infinity branches for every Point.
    let guard = next_up_finite(guard_scale * final_guard_factor);
    let lower = next_down_finite(center - quantization_error - guard);
    let upper = next_up_finite(center + quantization_error + guard);
    debug_assert!([center, lower, upper].into_iter().all(f64::is_finite));
    (lower, upper)
}

#[inline(always)]
pub(super) fn next_up_finite(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    let advanced = if bits >> 63 == 0 {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    };
    f64::from_bits(if magnitude == 0 { 1 } else { advanced })
}

#[inline(always)]
pub(super) fn next_down_finite(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    let advanced = if bits >> 63 == 0 {
        bits.wrapping_sub(1)
    } else {
        bits.wrapping_add(1)
    };
    f64::from_bits(if magnitude == 0 {
        0x8000_0000_0000_0001
    } else {
        advanced
    })
}

#[inline]
pub(super) fn add_up(left: f64, right: f64) -> f64 {
    if left == 0.0 {
        return right;
    }
    if right == 0.0 {
        return left;
    }
    (left + right).next_up()
}

#[inline(always)]
pub(super) fn mul_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    (left * right).next_up()
}

#[inline]
pub(super) fn sqrt_up(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value.sqrt().next_up()
    }
}

pub(super) fn f32_upper(value: f64) -> Result<f32, PerVectorScalarError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PerVectorScalarError::NonFinite);
    }
    let rounded = value as f32;
    if !rounded.is_finite() {
        return Err(PerVectorScalarError::NonFinite);
    }
    Ok(if f64::from(rounded) < value {
        rounded.next_up()
    } else {
        rounded
    })
}
