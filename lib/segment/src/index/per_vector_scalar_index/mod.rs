//! Production adapter for frozen Per-Vector Scalar Quantization V1.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Sequential;
use common::types::PointOffsetType;
use common::universal_io::{MmapFile, MmapFs};
use quantization::{
    DistanceType, EncodedStorage, EncodedStorageBuilder, EncodedVectorsPerVectorScalar,
    PerVectorScalarMetadata, VectorParameters, per_vector_scalar_row_bytes,
};

use super::native_dense_stream::{NativeDenseIndexCursor, NativeDensePlan};
use crate::common::check_stopped;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::named_vectors::CowVector;
use crate::data_types::vectors::VectorElementType;
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::quantized::quantized_chunked_mmap_storage::{
    QuantizedChunkedMmapStorage, QuantizedChunkedMmapStorageBuilder,
};
use crate::vector_storage::quantized::quantized_ram_storage::{
    QuantizedRamStorage, QuantizedRamStorageBuilder,
};
use crate::vector_storage::quantized::quantized_storage::{
    QuantizedStorage, QuantizedStorageBuilder,
};
use crate::vector_storage::raw_scorer::new_raw_scorer_preprocessed;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

pub(crate) const DATA_FILE: &str = "per_vector_scalar.data";
pub(crate) const METADATA_FILE: &str = "per_vector_scalar.meta.json";
const SCORE_CHUNK_SIZE: usize = 4_096;

pub struct PerVectorScalarIndex<TStorage: EncodedStorage> {
    encoded: EncodedVectorsPerVectorScalar<TStorage>,
}

pub type PerVectorScalarRamIndex = PerVectorScalarIndex<QuantizedRamStorage>;
pub type PerVectorScalarMmapIndex = PerVectorScalarIndex<QuantizedStorage<MmapFile>>;
pub type PerVectorScalarChunkedMmapIndex = PerVectorScalarIndex<QuantizedChunkedMmapStorage>;

mod storage;
use storage::validate_storage;

pub(crate) enum PerVectorScalarIndexVariant {
    Ram(PerVectorScalarRamIndex),
    Mmap(PerVectorScalarMmapIndex),
}

impl fmt::Debug for PerVectorScalarIndexVariant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PerVectorScalarIndexVariant")
            .field(&match self {
                Self::Ram(_) => "Ram",
                Self::Mmap(_) => "Mmap",
            })
            .finish()
    }
}

impl PerVectorScalarIndexVariant {
    pub(crate) fn cursor<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<NativeDenseIndexCursor<'a>> {
        match self {
            Self::Ram(index) => index.cursor(
                vector_storage,
                eligible,
                raw_query,
                hardware_counter,
                stopped,
            ),
            Self::Mmap(index) => index.cursor(
                vector_storage,
                eligible,
                raw_query,
                hardware_counter,
                stopped,
            ),
        }
    }

    pub(crate) fn valid_for(&self, vector_count: usize, dimension: usize) -> bool {
        let metadata = match self {
            Self::Ram(index) => index.encoded.metadata(),
            Self::Mmap(index) => index.encoded.metadata(),
        };
        metadata.vector_count() == vector_count && metadata.dimension() == dimension
    }

    pub(crate) fn files(&self) -> Vec<PathBuf> {
        let mut files = match self {
            Self::Ram(index) => index.encoded.storage().files(),
            Self::Mmap(index) => index.encoded.storage().files(),
        };
        if let Some(data_path) = files.first() {
            files.push(data_path.with_file_name(METADATA_FILE));
        }
        files
    }

    pub(crate) fn heap_size_bytes(&self) -> usize {
        match self {
            Self::Ram(index) => index.encoded.storage().heap_size_bytes(),
            Self::Mmap(index) => index.encoded.storage().heap_size_bytes(),
        }
    }

    pub(crate) fn populate(&self) {
        if let Self::Mmap(index) = self {
            index.encoded.storage().populate();
        }
    }

    pub(crate) fn clear_cache(&self) {
        if let Self::Mmap(index) = self {
            index.encoded.storage().clear_cache();
        }
    }
}

impl<TStorage: EncodedStorage> PerVectorScalarIndex<TStorage> {
    fn cursor<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        mut eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<NativeDenseIndexCursor<'a>> {
        check_stopped(stopped)?;
        validate_storage(vector_storage, self.encoded.metadata().dimension())?;
        if vector_storage.total_vector_count() != self.encoded.metadata().vector_count() {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar index does not match the frozen Segment vector count",
            ));
        }
        if raw_query.len() != self.encoded.metadata().dimension()
            || raw_query.iter().any(|coordinate| !coordinate.is_finite())
        {
            return Err(OperationError::validation_error(
                "PerVectorScalar cursor requires a finite Query of matching dimension",
            ));
        }
        eligible.sort_unstable();
        if eligible.windows(2).any(|pair| pair[0] == pair[1])
            || eligible
                .iter()
                .any(|id| *id as usize >= self.encoded.metadata().vector_count())
        {
            return Err(OperationError::inconsistent_storage(
                "PerVectorScalar eligible universe is invalid",
            ));
        }

        let prepared_query = vector_storage
            .distance()
            .preprocess_vector::<VectorElementType>(raw_query.to_vec());
        let encoded_query = self
            .encoded
            .try_encode_query(&prepared_query)
            .map_err(|error| {
                OperationError::validation_error(format!(
                    "failed to encode PerVectorScalar Query: {error}",
                ))
            })?;
        let guard = compact_style_guard_factor(prepared_query.len()).ok_or_else(|| {
            OperationError::validation_error(
                "PerVectorScalar floating guard is unsupported for this dimension",
            )
        })?;
        let mut bounds = Vec::with_capacity(eligible.len());
        let contiguous = eligible.len() == self.encoded.metadata().vector_count()
            && eligible
                .iter()
                .enumerate()
                .all(|(ordinal, id)| *id as usize == ordinal);
        if contiguous {
            for start in (0..eligible.len()).step_by(SCORE_CHUNK_SIZE) {
                check_stopped(stopped)?;
                let count = SCORE_CHUNK_SIZE.min(eligible.len() - start);
                self.encoded
                    .for_each_final_certified_range(
                        &encoded_query,
                        start as PointOffsetType,
                        count,
                        guard,
                        hardware_counter,
                        |point_id, lower, upper| bounds.push((point_id, lower, upper)),
                    )
                    .map_err(|error| {
                        OperationError::inconsistent_storage(format!(
                            "PerVectorScalar certificate failed: {error}",
                        ))
                    })?;
            }
        } else {
            for chunk in eligible.chunks(SCORE_CHUNK_SIZE) {
                check_stopped(stopped)?;
                self.encoded
                    .for_each_final_certified_batch(
                        &encoded_query,
                        chunk,
                        guard,
                        hardware_counter,
                        |point_id, lower, upper| bounds.push((point_id, lower, upper)),
                    )
                    .map_err(|error| {
                        OperationError::inconsistent_storage(format!(
                            "PerVectorScalar certificate failed: {error}",
                        ))
                    })?;
            }
        }

        let point_count = eligible.len();
        let exact_scorer =
            new_raw_scorer_preprocessed(prepared_query, vector_storage, hardware_counter.fork())?;
        NativeDenseIndexCursor::from_certificate_bounds(
            eligible,
            bounds,
            move |id| exact_scorer.score_point(id),
            NativeDensePlan::PerVectorScalarCertificate,
            point_count,
        )
    }
}

fn compact_style_guard_factor(dimension: usize) -> Option<f64> {
    let operations = dimension.checked_mul(8)?.checked_add(64)? as f64;
    let numerator = (operations * f64::from(f32::EPSILON)).next_up();
    if numerator >= 1.0 {
        return None;
    }
    Some((numerator / (1.0 - numerator).next_down()).next_up())
}

#[cfg(test)]
mod tests;
