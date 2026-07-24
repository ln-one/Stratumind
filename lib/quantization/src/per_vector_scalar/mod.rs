//! Frozen per-vector signed-i8 quantization V1 with deterministic bounds.
//!
//! This crate layer deliberately knows nothing about Segment metrics or
//! `RawScorer`. It bounds the mathematical dot product of already-preprocessed
//! f32 coordinates. The Segment layer is responsible for adding the
//! architecture-specific authoritative-f32 scorer guard.

mod simd;

use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use common::counter::hardware_counter::HardwareCounterCell;
use common::fs::atomic_save_json;
use common::mmap::MmapFlusher;
use common::typelevel::True;
use common::types::PointOffsetType;
use fs_err as fs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use self::simd::{dot_i8, resolve_dot_i8, resolve_dot_i8_x4, selected_kernel_name};
use crate::EncodingError;
use crate::encoded_storage::{EncodedStorage, EncodedStorageBuilder, validate_storage_vector_size};
use crate::encoded_vectors::{
    DistanceType, EncodedVectors, VectorParameters, validate_vector_parameters,
};

const MAGIC: &str = "QPVSI8";
const FORMAT_VERSION: u32 = 1;
const FLOATING_BOUND_VERSION: u32 = 2;
const HEADER_BYTES: usize = 4 * std::mem::size_of::<f32>();
const ALIGNMENT: usize = 16;
const MAX_CODE: f64 = 127.0;
const CHECKSUM_ALGORITHM: &str = "sha256";
// The certified hot loop contains no floating-point coordinate reduction: the
// signed-i8 dot product is exact in i32 and is converted exactly to f64. The
// remaining center/radius expression has fewer than eight rounded f64
// operations. Sixty-four epsilons leave a deliberately wide margin around
// their standard gamma_n error while remaining negligible beside the f32
// quantization residual.
const CERTIFICATE_F64_GUARD: f64 = 64.0 * f64::EPSILON;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerVectorScalarBounds {
    pub point_id: PointOffsetType,
    pub center: f64,
    pub document_original_norm_upper: f64,
    /// Lower bound around the mathematical f64 dot product of the frozen f32
    /// coordinates. It does not yet include the authoritative RawScorer guard.
    pub quantization_lower: f64,
    /// Upper bound around the mathematical f64 dot product of the frozen f32
    /// coordinates. It does not yet include the authoritative RawScorer guard.
    pub quantization_upper: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerVectorScalarRowStats {
    pub scale: f32,
    pub original_norm_upper: f32,
    pub reconstructed_norm_upper: f32,
    pub residual_norm_upper: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedQueryPerVectorScalar {
    scale: f32,
    codes: Vec<i8>,
    original_norm_upper: f64,
    reconstructed_norm_upper: f64,
    residual_norm_upper: f64,
}

impl EncodedQueryPerVectorScalar {
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    pub fn original_norm_upper(&self) -> f64 {
        self.original_norm_upper
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PerVectorScalarMetadata {
    magic: String,
    format_version: u32,
    dimension: usize,
    actual_dimension: usize,
    row_bytes: usize,
    vector_count: usize,
    data_bytes: usize,
    endianness: String,
    code_encoding: String,
    floating_bound_version: u32,
    source_generation: u64,
    data_checksum_algorithm: String,
    data_checksum: [u8; 32],
    vector_parameters: VectorParameters,
}

impl PerVectorScalarMetadata {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn actual_dimension(&self) -> usize {
        self.actual_dimension
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn vector_count(&self) -> usize {
        self.vector_count
    }

    pub fn source_generation(&self) -> u64 {
        self.source_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerVectorScalarError {
    InvalidQuery,
    InvalidRow,
    NonFinite,
    UnsupportedDistance,
    Overflow,
}

impl Display for PerVectorScalarError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidQuery => formatter.write_str("invalid PerVectorScalar query"),
            Self::InvalidRow => formatter.write_str("invalid PerVectorScalar encoded row"),
            Self::NonFinite => formatter.write_str("PerVectorScalar produced a non-finite value"),
            Self::UnsupportedDistance => {
                formatter.write_str("PerVectorScalar supports non-inverted Dot/Cosine only")
            }
            Self::Overflow => formatter.write_str("PerVectorScalar size arithmetic overflow"),
        }
    }
}

impl std::error::Error for PerVectorScalarError {}

pub struct EncodedVectorsPerVectorScalar<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: PerVectorScalarMetadata,
    metadata_path: Option<PathBuf>,
}

impl<TStorage: EncodedStorage> EncodedVectorsPerVectorScalar<TStorage> {
    pub fn selected_kernel_name(&self) -> &'static str {
        let contiguous = self
            .encoded_vectors
            .get_contiguous_vector_data(0, 1)
            .is_some();
        selected_kernel_name(self.metadata.actual_dimension, contiguous)
    }

    pub fn storage_residency(&self) -> &'static str {
        if self.encoded_vectors.is_on_disk() {
            "mmap"
        } else {
            "RAM"
        }
    }
}

mod bounds;
mod format;
mod qdrant;
mod scan;

pub use format::per_vector_scalar_row_bytes;

#[cfg(all(test, feature = "testing"))]
mod tests;
