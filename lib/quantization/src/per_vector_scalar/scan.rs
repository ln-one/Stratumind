use super::bounds::*;
use super::format::{parse_row, parse_validated_row};
use super::*;

impl<TStorage: EncodedStorage> EncodedVectorsPerVectorScalar<TStorage> {
    pub fn score_certified_batch(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        bounds_out: &mut Vec<PerVectorScalarBounds>,
        hardware_counter: &HardwareCounterCell,
    ) -> Result<(), PerVectorScalarError> {
        bounds_out.clear();
        bounds_out.reserve(ids.len());
        self.for_each_certified_batch(query, ids, hardware_counter, |bound| {
            bounds_out.push(bound);
        })
    }

    pub fn for_each_certified_batch(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        hardware_counter: &HardwareCounterCell,
        mut consume: impl FnMut(PerVectorScalarBounds),
    ) -> Result<(), PerVectorScalarError> {
        if query.codes.len() != self.metadata.actual_dimension {
            return Err(PerVectorScalarError::InvalidQuery);
        }
        hardware_counter
            .vector_io_read()
            .incr_delta(self.metadata.row_bytes.saturating_mul(ids.len()));
        hardware_counter
            .cpu_counter()
            .incr_delta(self.metadata.dimension.saturating_mul(ids.len()));

        if ids
            .windows(2)
            .all(|pair| pair[1] == pair[0].saturating_add(1))
            && let Some(dot_x4) = resolve_dot_i8_x4(self.metadata.actual_dimension)
            && self.for_each_contiguous_certified_batch_x4(query, ids, dot_x4, &mut consume)?
        {
            return Ok(());
        }

        let dot = resolve_dot_i8(self.metadata.actual_dimension);
        self.for_each_certified_batch_with_dot(query, ids, dot, consume)
    }

    /// Research-only FastScan-style final handler. It combines the
    /// quantization and authoritative-f32 error terms before constructing the
    /// two directed endpoints, avoiding an intermediate certificate per Point.
    ///
    /// `final_guard_factor` is the dimension-dependent factor used by Qdrant's
    /// mature Compact certificate. The per-Point scale includes both norms,
    /// the approximation center and the quantization residual.
    pub fn for_each_final_certified_batch(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        final_guard_factor: f64,
        hardware_counter: &HardwareCounterCell,
        mut consume: impl FnMut(PointOffsetType, f64, f64),
    ) -> Result<(), PerVectorScalarError> {
        if query.codes.len() != self.metadata.actual_dimension
            || !final_guard_factor.is_finite()
            || final_guard_factor < 0.0
        {
            return Err(PerVectorScalarError::InvalidQuery);
        }
        hardware_counter
            .vector_io_read()
            .incr_delta(self.metadata.row_bytes.saturating_mul(ids.len()));
        hardware_counter
            .cpu_counter()
            .incr_delta(self.metadata.dimension.saturating_mul(ids.len()));

        if ids
            .windows(2)
            .all(|pair| pair[1] == pair[0].saturating_add(1))
            && let Some(dot_x4) = resolve_dot_i8_x4(self.metadata.actual_dimension)
            && self.for_each_contiguous_final_certified_batch_x4(
                query,
                ids,
                dot_x4,
                final_guard_factor,
                &mut consume,
            )?
        {
            return Ok(());
        }

        let dot = resolve_dot_i8(self.metadata.actual_dimension);
        let mut consumed = 0usize;
        for (ordinal, row) in self.encoded_vectors.iter_batch(ids) {
            let point_id = *ids.get(ordinal).ok_or(PerVectorScalarError::InvalidRow)?;
            let (stats, codes) = parse_validated_row(&row, &self.metadata)?;
            let (lower, upper) = final_bounds_from_integer_dot(
                query,
                stats,
                dot(query.codes(), codes),
                final_guard_factor,
            );
            consume(point_id, lower, upper);
            consumed += 1;
        }
        if consumed != ids.len() {
            return Err(PerVectorScalarError::InvalidRow);
        }
        Ok(())
    }

    /// Scores one contiguous identity range without first materializing its
    /// point IDs. This is the no-filter immutable-Segment fast path; storages
    /// without a contiguous range view retain the exact point-at-a-time
    /// fallback.
    pub fn for_each_final_certified_range(
        &self,
        query: &EncodedQueryPerVectorScalar,
        start: PointOffsetType,
        count: usize,
        final_guard_factor: f64,
        hardware_counter: &HardwareCounterCell,
        mut consume: impl FnMut(PointOffsetType, f64, f64),
    ) -> Result<(), PerVectorScalarError> {
        let start = start as usize;
        let end = start
            .checked_add(count)
            .ok_or(PerVectorScalarError::Overflow)?;
        if query.codes.len() != self.metadata.actual_dimension
            || !final_guard_factor.is_finite()
            || final_guard_factor < 0.0
            || end > self.metadata.vector_count
            || end > PointOffsetType::MAX as usize + 1
        {
            return Err(PerVectorScalarError::InvalidQuery);
        }
        hardware_counter
            .vector_io_read()
            .incr_delta(self.metadata.row_bytes.saturating_mul(count));
        hardware_counter
            .cpu_counter()
            .incr_delta(self.metadata.dimension.saturating_mul(count));

        if let Some(dot_x4) = resolve_dot_i8_x4(self.metadata.actual_dimension) {
            let dot_tail = resolve_dot_i8(self.metadata.actual_dimension);
            let mut consumed = 0usize;
            while consumed < count {
                let point_id = PointOffsetType::try_from(start + consumed)
                    .map_err(|_| PerVectorScalarError::Overflow)?;
                let Some((contiguous_count, rows)) = self
                    .encoded_vectors
                    .get_contiguous_vector_data(point_id, count - consumed)
                else {
                    if consumed == 0 {
                        break;
                    }
                    return Err(PerVectorScalarError::InvalidRow);
                };
                consume_contiguous_rows_x4_final(
                    query,
                    point_id,
                    contiguous_count,
                    &rows,
                    &self.metadata,
                    dot_x4,
                    dot_tail,
                    final_guard_factor,
                    &mut consume,
                )?;
                consumed += contiguous_count;
            }
            if consumed == count {
                return Ok(());
            }
        }

        let dot = resolve_dot_i8(self.metadata.actual_dimension);
        for ordinal in start..end {
            let point_id =
                PointOffsetType::try_from(ordinal).map_err(|_| PerVectorScalarError::Overflow)?;
            let row = self.encoded_vectors.get_vector_data(point_id);
            let (stats, codes) = parse_validated_row(&row, &self.metadata)?;
            let (lower, upper) = final_bounds_from_integer_dot(
                query,
                stats,
                dot(query.codes(), codes),
                final_guard_factor,
            );
            consume(point_id, lower, upper);
        }
        Ok(())
    }

    fn for_each_contiguous_certified_batch_x4(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        dot_x4: simd::DotI8x4Fn,
        consume: &mut impl FnMut(PerVectorScalarBounds),
    ) -> Result<bool, PerVectorScalarError> {
        if ids.is_empty() {
            return Ok(true);
        }
        let dot_tail = resolve_dot_i8(self.metadata.actual_dimension);
        let mut consumed = 0usize;
        while consumed < ids.len() {
            let start = ids[consumed];
            let Some((count, rows)) = self
                .encoded_vectors
                .get_contiguous_vector_data(start, ids.len() - consumed)
            else {
                if consumed == 0 {
                    return Ok(false);
                }
                return Err(PerVectorScalarError::InvalidRow);
            };
            consume_contiguous_rows_x4(
                query,
                ids,
                consumed,
                count,
                &rows,
                &self.metadata,
                dot_x4,
                dot_tail,
                consume,
            )?;
            consumed += count;
        }
        Ok(true)
    }

    fn for_each_contiguous_final_certified_batch_x4(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        dot_x4: simd::DotI8x4Fn,
        final_guard_factor: f64,
        consume: &mut impl FnMut(PointOffsetType, f64, f64),
    ) -> Result<bool, PerVectorScalarError> {
        if ids.is_empty() {
            return Ok(true);
        }
        let dot_tail = resolve_dot_i8(self.metadata.actual_dimension);
        let mut consumed = 0usize;
        while consumed < ids.len() {
            let start = ids[consumed];
            let Some((count, rows)) = self
                .encoded_vectors
                .get_contiguous_vector_data(start, ids.len() - consumed)
            else {
                if consumed == 0 {
                    return Ok(false);
                }
                return Err(PerVectorScalarError::InvalidRow);
            };
            consume_contiguous_rows_x4_final(
                query,
                start,
                count,
                &rows,
                &self.metadata,
                dot_x4,
                dot_tail,
                final_guard_factor,
                consume,
            )?;
            consumed += count;
        }
        Ok(true)
    }

    fn for_each_certified_batch_with_dot<D>(
        &self,
        query: &EncodedQueryPerVectorScalar,
        ids: &[PointOffsetType],
        dot: D,
        mut consume: impl FnMut(PerVectorScalarBounds),
    ) -> Result<(), PerVectorScalarError>
    where
        D: Copy + Fn(&[i8], &[i8]) -> i64,
    {
        let mut consumed = 0usize;
        for (ordinal, row) in self.encoded_vectors.iter_batch(ids) {
            let point_id = *ids.get(ordinal).ok_or(PerVectorScalarError::InvalidRow)?;
            // Encode/load validates every immutable row once. Keep only the
            // structural length check in the scoring hot path.
            let (stats, codes) = parse_validated_row(&row, &self.metadata)?;
            consume(quantization_bounds(point_id, query, stats, codes, dot)?);
            consumed += 1;
        }
        if consumed != ids.len() {
            return Err(PerVectorScalarError::InvalidRow);
        }
        Ok(())
    }

    pub(super) fn center(
        &self,
        query: &EncodedQueryPerVectorScalar,
        row: &[u8],
    ) -> Result<f64, PerVectorScalarError> {
        let (stats, codes) = parse_row(row, &self.metadata)?;
        center(query, stats, codes)
    }

    pub(super) fn center_point(
        &self,
        query: &EncodedQueryPerVectorScalar,
        row: &[u8],
    ) -> Result<f64, PerVectorScalarError> {
        let (stats, codes) = parse_validated_row(row, &self.metadata)?;
        center(query, stats, codes)
    }
}

#[expect(clippy::too_many_arguments)]
fn consume_contiguous_rows_x4(
    query: &EncodedQueryPerVectorScalar,
    ids: &[PointOffsetType],
    ordinal: usize,
    count: usize,
    rows: &[u8],
    metadata: &PerVectorScalarMetadata,
    dot_x4: simd::DotI8x4Fn,
    dot_tail: simd::DotI8Fn,
    consume: &mut impl FnMut(PerVectorScalarBounds),
) -> Result<(), PerVectorScalarError> {
    let row_bytes = metadata.row_bytes;
    if count == 0 || rows.len() != count.saturating_mul(row_bytes) {
        return Err(PerVectorScalarError::InvalidRow);
    }

    let mut offset = 0usize;
    while offset + 4 <= count {
        let row = |index: usize| {
            let start = (offset + index) * row_bytes;
            &rows[start..start + row_bytes]
        };
        let (stats_0, codes_0) = parse_validated_row(row(0), metadata)?;
        let (stats_1, codes_1) = parse_validated_row(row(1), metadata)?;
        let (stats_2, codes_2) = parse_validated_row(row(2), metadata)?;
        let (stats_3, codes_3) = parse_validated_row(row(3), metadata)?;
        let dots = dot_x4(query.codes(), [codes_0, codes_1, codes_2, codes_3]);
        for (index, (stats, integer_dot)) in [
            (stats_0, dots[0]),
            (stats_1, dots[1]),
            (stats_2, dots[2]),
            (stats_3, dots[3]),
        ]
        .into_iter()
        .enumerate()
        {
            let point_id = *ids
                .get(ordinal + offset + index)
                .ok_or(PerVectorScalarError::InvalidRow)?;
            consume(quantization_bounds_from_integer_dot(
                point_id,
                query,
                stats,
                integer_dot,
            )?);
        }
        offset += 4;
    }

    while offset < count {
        let start = offset * row_bytes;
        let row = &rows[start..start + row_bytes];
        let (stats, codes) = parse_validated_row(row, metadata)?;
        let point_id = *ids
            .get(ordinal + offset)
            .ok_or(PerVectorScalarError::InvalidRow)?;
        consume(quantization_bounds_from_integer_dot(
            point_id,
            query,
            stats,
            dot_tail(query.codes(), codes),
        )?);
        offset += 1;
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn consume_contiguous_rows_x4_final(
    query: &EncodedQueryPerVectorScalar,
    first_point_id: PointOffsetType,
    count: usize,
    rows: &[u8],
    metadata: &PerVectorScalarMetadata,
    dot_x4: simd::DotI8x4Fn,
    dot_tail: simd::DotI8Fn,
    final_guard_factor: f64,
    consume: &mut impl FnMut(PointOffsetType, f64, f64),
) -> Result<(), PerVectorScalarError> {
    let row_bytes = metadata.row_bytes;
    if count == 0 || rows.len() != count.saturating_mul(row_bytes) {
        return Err(PerVectorScalarError::InvalidRow);
    }

    let mut offset = 0usize;
    while offset + 4 <= count {
        let row = |index: usize| {
            let start = (offset + index) * row_bytes;
            &rows[start..start + row_bytes]
        };
        let (stats_0, codes_0) = parse_validated_row(row(0), metadata)?;
        let (stats_1, codes_1) = parse_validated_row(row(1), metadata)?;
        let (stats_2, codes_2) = parse_validated_row(row(2), metadata)?;
        let (stats_3, codes_3) = parse_validated_row(row(3), metadata)?;
        let dots = dot_x4(query.codes(), [codes_0, codes_1, codes_2, codes_3]);
        for (index, (stats, integer_dot)) in [
            (stats_0, dots[0]),
            (stats_1, dots[1]),
            (stats_2, dots[2]),
            (stats_3, dots[3]),
        ]
        .into_iter()
        .enumerate()
        {
            let point_id = PointOffsetType::try_from(first_point_id as usize + offset + index)
                .map_err(|_| PerVectorScalarError::Overflow)?;
            let (lower, upper) =
                final_bounds_from_integer_dot(query, stats, integer_dot, final_guard_factor);
            consume(point_id, lower, upper);
        }
        offset += 4;
    }

    while offset < count {
        let start = offset * row_bytes;
        let row = &rows[start..start + row_bytes];
        let (stats, codes) = parse_validated_row(row, metadata)?;
        let point_id = PointOffsetType::try_from(first_point_id as usize + offset)
            .map_err(|_| PerVectorScalarError::Overflow)?;
        let (lower, upper) = final_bounds_from_integer_dot(
            query,
            stats,
            dot_tail(query.codes(), codes),
            final_guard_factor,
        );
        consume(point_id, lower, upper);
        offset += 1;
    }
    Ok(())
}

fn center(
    query: &EncodedQueryPerVectorScalar,
    stats: PerVectorScalarRowStats,
    codes: &[i8],
) -> Result<f64, PerVectorScalarError> {
    let integer_dot = dot_i8(&query.codes, codes);
    let center = integer_dot as f64 * f64::from(query.scale) * f64::from(stats.scale);
    center
        .is_finite()
        .then_some(center)
        .ok_or(PerVectorScalarError::NonFinite)
}
