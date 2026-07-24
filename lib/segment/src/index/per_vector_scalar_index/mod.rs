//! Segment integration for frozen Per-Vector Scalar Quantization V1.
//!
//! Production owns the V1 storage and exact cursor. Alternative encodings and
//! historical ablations remain behind the research feature.

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

use super::exact_dense_stream::{
    DenseExecutionPolicy, DensePhysicalPlan, ExactDenseCursor, PendingBound,
};
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cursor_with_refine_batch<'a>(
        &self,
        vector_storage: &'a VectorStorageEnum,
        eligible: Vec<PointOffsetType>,
        raw_query: &[f32],
        exact_refine_batch: usize,
        hardware_counter: &HardwareCounterCell,
        stopped: &AtomicBool,
    ) -> OperationResult<ExactDenseCursor<'a>> {
        match self {
            Self::Ram(index) => {
                if is_complete_contiguous_universe(index, &eligible) {
                    index.cursor_contiguous_with_refine_batch(
                        vector_storage,
                        raw_query,
                        exact_refine_batch,
                        hardware_counter,
                        stopped,
                    )
                } else {
                    index.cursor_with_refine_batch(
                        vector_storage,
                        eligible,
                        raw_query,
                        exact_refine_batch,
                        hardware_counter,
                        stopped,
                    )
                }
            }
            Self::Mmap(index) => {
                if is_complete_contiguous_universe(index, &eligible) {
                    index.cursor_contiguous_with_refine_batch(
                        vector_storage,
                        raw_query,
                        exact_refine_batch,
                        hardware_counter,
                        stopped,
                    )
                } else {
                    index.cursor_with_refine_batch(
                        vector_storage,
                        eligible,
                        raw_query,
                        exact_refine_batch,
                        hardware_counter,
                        stopped,
                    )
                }
            }
        }
    }

    pub(crate) fn valid_for(&self, vector_count: usize, dimension: usize) -> bool {
        let metadata = match self {
            Self::Ram(index) => index.encoded().metadata(),
            Self::Mmap(index) => index.encoded().metadata(),
        };
        metadata.vector_count() == vector_count && metadata.dimension() == dimension
    }

    pub(crate) fn files(&self) -> Vec<PathBuf> {
        let mut files = match self {
            Self::Ram(index) => index.encoded().storage().files(),
            Self::Mmap(index) => index.encoded().storage().files(),
        };
        if let Some(data_path) = files.first() {
            files.push(data_path.with_file_name(METADATA_FILE));
        }
        files
    }

    pub(crate) fn heap_size_bytes(&self) -> usize {
        match self {
            Self::Ram(index) => index.encoded().storage().heap_size_bytes(),
            Self::Mmap(index) => index.encoded().storage().heap_size_bytes(),
        }
    }

    pub(crate) fn populate(&self) {
        if let Self::Mmap(index) = self {
            index.encoded().storage().populate();
        }
    }

    pub(crate) fn clear_cache(&self) {
        if let Self::Mmap(index) = self {
            index.encoded().storage().clear_cache();
        }
    }
}

fn is_complete_contiguous_universe<TStorage: EncodedStorage>(
    index: &PerVectorScalarIndex<TStorage>,
    eligible: &[PointOffsetType],
) -> bool {
    eligible.len() == index.encoded().metadata().vector_count()
        && eligible
            .iter()
            .enumerate()
            .all(|(ordinal, id)| *id as usize == ordinal)
}

mod contiguous;
mod cursor;
mod storage;

#[cfg(feature = "stratumind-research")]
pub use cursor::PerVectorScalarCursorProfile;

#[cfg(test)]
mod production_tests;
#[cfg(all(test, feature = "stratumind-research"))]
mod tests;
