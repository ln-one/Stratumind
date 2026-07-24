use super::bounds::*;
use super::*;

impl<TStorage: EncodedStorage> EncodedVectorsPerVectorScalar<TStorage> {
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'a>(
        orig_data: impl Iterator<Item = impl AsRef<[f32]> + 'a> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<Storage = TStorage>,
        vector_parameters: &VectorParameters,
        count: usize,
        source_generation: u64,
        meta_path: Option<&Path>,
        stopped: &AtomicBool,
    ) -> Result<Self, EncodingError> {
        validate_supported_parameters(vector_parameters).map_err(to_encoding_error)?;
        validate_vector_parameters(orig_data.clone(), vector_parameters)?;

        let actual_dimension = aligned_dimension(vector_parameters.dim);
        let row_bytes = HEADER_BYTES
            .checked_add(actual_dimension)
            .ok_or_else(|| to_encoding_error(PerVectorScalarError::Overflow))?;
        let data_bytes = count
            .checked_mul(row_bytes)
            .ok_or_else(|| to_encoding_error(PerVectorScalarError::Overflow))?;
        let mut checksum = Sha256::new();
        let mut encoded_count = 0usize;

        for vector in orig_data {
            if stopped.load(Ordering::Relaxed) {
                return Err(EncodingError::Stopped);
            }
            let row = encode_row(vector.as_ref(), vector_parameters.dim, actual_dimension)
                .map_err(to_encoding_error)?;
            checksum.update(&row);
            storage_builder.push_vector_data(&row).map_err(|error| {
                EncodingError::EncodingError(
                    format!("failed to push PerVectorScalar row: {error}",),
                )
            })?;
            encoded_count += 1;
        }
        if encoded_count != count {
            return Err(EncodingError::ArgumentsError(format!(
                "PerVectorScalar expected {count} vectors but encoded {encoded_count}",
            )));
        }

        let encoded_vectors = storage_builder.build().map_err(|error| {
            EncodingError::EncodingError(format!(
                "failed to build PerVectorScalar storage: {error}",
            ))
        })?;
        let metadata = PerVectorScalarMetadata {
            magic: MAGIC.to_owned(),
            format_version: FORMAT_VERSION,
            dimension: vector_parameters.dim,
            actual_dimension,
            row_bytes,
            vector_count: count,
            data_bytes,
            endianness: "little".to_owned(),
            code_encoding: "signed_i8_twos_complement".to_owned(),
            floating_bound_version: FLOATING_BOUND_VERSION,
            source_generation,
            data_checksum_algorithm: CHECKSUM_ALGORITHM.to_owned(),
            data_checksum: checksum.finalize().into(),
            vector_parameters: *vector_parameters,
        };

        if let Some(meta_path) = meta_path {
            let parent = meta_path.parent().ok_or_else(|| {
                EncodingError::IOError("metadata path must have a parent".to_owned())
            })?;
            fs::create_dir_all(parent).map_err(|error| {
                EncodingError::IOError(format!(
                    "failed to create PerVectorScalar metadata directory: {error}",
                ))
            })?;
            atomic_save_json(meta_path, &metadata).map_err(|error| {
                EncodingError::IOError(format!(
                    "failed to atomically publish PerVectorScalar metadata: {error}",
                ))
            })?;
        }

        Ok(Self {
            encoded_vectors,
            metadata,
            metadata_path: meta_path.map(PathBuf::from),
        })
    }

    pub fn load(encoded_vectors: TStorage, meta_path: &Path) -> std::io::Result<Self> {
        let contents = fs::read_to_string(meta_path)?;
        let metadata: PerVectorScalarMetadata = serde_json::from_str(&contents)?;
        validate_metadata(&metadata, &encoded_vectors)?;

        let mut checksum = Sha256::new();
        for index in 0..metadata.vector_count {
            let row = encoded_vectors.get_vector_data(index as PointOffsetType);
            validate_row(&row, &metadata).map_err(invalid_data)?;
            checksum.update(&row);
        }
        let actual_checksum: [u8; 32] = checksum.finalize().into();
        if actual_checksum != metadata.data_checksum {
            return Err(invalid_data("PerVectorScalar data checksum mismatch"));
        }

        Ok(Self {
            encoded_vectors,
            metadata,
            metadata_path: Some(meta_path.to_path_buf()),
        })
    }

    pub fn metadata(&self) -> &PerVectorScalarMetadata {
        &self.metadata
    }

    pub fn storage(&self) -> &TStorage {
        &self.encoded_vectors
    }

    pub fn try_encode_query(
        &self,
        query: &[f32],
    ) -> Result<EncodedQueryPerVectorScalar, PerVectorScalarError> {
        if query.len() != self.metadata.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(PerVectorScalarError::InvalidQuery);
        }
        encode_query(query, self.metadata.actual_dimension)
    }
}

pub(super) fn validate_supported_parameters(
    vector_parameters: &VectorParameters,
) -> Result<(), PerVectorScalarError> {
    if vector_parameters.dim == 0
        || vector_parameters.invert
        || !matches!(
            vector_parameters.distance_type,
            DistanceType::Dot | DistanceType::Cosine
        )
    {
        return Err(PerVectorScalarError::UnsupportedDistance);
    }
    Ok(())
}

pub(super) fn aligned_dimension(dimension: usize) -> usize {
    dimension + (ALIGNMENT - dimension % ALIGNMENT) % ALIGNMENT
}

pub fn per_vector_scalar_row_bytes(dimension: usize) -> Option<usize> {
    HEADER_BYTES.checked_add(aligned_dimension(dimension))
}

pub(super) fn encode_row(
    vector: &[f32],
    dimension: usize,
    actual_dimension: usize,
) -> Result<Vec<u8>, PerVectorScalarError> {
    if vector.len() != dimension || vector.iter().any(|value| !value.is_finite()) {
        return Err(PerVectorScalarError::NonFinite);
    }
    let (scale, codes, stats) = encode_coordinates(vector, actual_dimension)?;
    let mut row = Vec::with_capacity(HEADER_BYTES + actual_dimension);
    row.extend_from_slice(&scale.to_le_bytes());
    row.extend_from_slice(&stats.original_norm_upper.to_le_bytes());
    row.extend_from_slice(&stats.reconstructed_norm_upper.to_le_bytes());
    row.extend_from_slice(&stats.residual_norm_upper.to_le_bytes());
    row.extend(codes.into_iter().map(|code| code as u8));
    Ok(row)
}

pub(super) fn encode_query(
    query: &[f32],
    actual_dimension: usize,
) -> Result<EncodedQueryPerVectorScalar, PerVectorScalarError> {
    let (scale, codes, stats) = encode_coordinates(query, actual_dimension)?;
    Ok(EncodedQueryPerVectorScalar {
        scale,
        codes,
        original_norm_upper: f64::from(stats.original_norm_upper),
        reconstructed_norm_upper: f64::from(stats.reconstructed_norm_upper),
        residual_norm_upper: f64::from(stats.residual_norm_upper),
    })
}

pub(super) fn encode_coordinates(
    vector: &[f32],
    actual_dimension: usize,
) -> Result<(f32, Vec<i8>, PerVectorScalarRowStats), PerVectorScalarError> {
    let max_abs = vector.iter().map(|value| value.abs()).fold(0.0, f32::max);
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        let candidate = (f64::from(max_abs) / MAX_CODE) as f32;
        if candidate == 0.0 {
            f32::from_bits(1)
        } else {
            candidate
        }
    };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PerVectorScalarError::NonFinite);
    }

    let mut codes = Vec::with_capacity(actual_dimension);
    for &value in vector {
        let code = (f64::from(value) / f64::from(scale))
            .round()
            .clamp(-MAX_CODE, MAX_CODE) as i8;
        codes.push(code);
    }
    codes.resize(actual_dimension, 0);

    let mut original_squared = 0.0;
    let mut reconstructed_squared = 0.0;
    let mut residual_squared = 0.0;
    for (&value, &code) in vector.iter().zip(&codes) {
        let original = f64::from(value);
        let reconstructed = f64::from(code) * f64::from(scale);
        let residual = original - reconstructed;
        original_squared = add_up(original_squared, mul_up(original.abs(), original.abs()));
        reconstructed_squared = add_up(
            reconstructed_squared,
            mul_up(reconstructed.abs(), reconstructed.abs()),
        );
        residual_squared = add_up(residual_squared, mul_up(residual.abs(), residual.abs()));
    }

    let stats = PerVectorScalarRowStats {
        scale,
        original_norm_upper: f32_upper(sqrt_up(original_squared))?,
        reconstructed_norm_upper: f32_upper(sqrt_up(reconstructed_squared))?,
        residual_norm_upper: f32_upper(sqrt_up(residual_squared))?,
    };
    Ok((scale, codes, stats))
}

pub(super) fn validate_metadata(
    metadata: &PerVectorScalarMetadata,
    storage: &impl EncodedStorage,
) -> std::io::Result<()> {
    if metadata.magic != MAGIC
        || metadata.format_version != FORMAT_VERSION
        || metadata.floating_bound_version != FLOATING_BOUND_VERSION
        || metadata.endianness != "little"
        || metadata.code_encoding != "signed_i8_twos_complement"
        || metadata.data_checksum_algorithm != CHECKSUM_ALGORITHM
    {
        return Err(invalid_data(
            "unsupported or corrupt PerVectorScalar metadata",
        ));
    }
    validate_supported_parameters(&metadata.vector_parameters).map_err(invalid_data)?;
    if metadata.dimension != metadata.vector_parameters.dim
        || metadata.actual_dimension != aligned_dimension(metadata.dimension)
        || metadata.row_bytes != HEADER_BYTES + metadata.actual_dimension
        || metadata.vector_count != storage.vectors_count()
        || metadata.data_bytes
            != metadata
                .row_bytes
                .checked_mul(metadata.vector_count)
                .ok_or_else(|| invalid_data("PerVectorScalar metadata size overflow"))?
    {
        return Err(invalid_data(
            "PerVectorScalar metadata does not match encoded storage",
        ));
    }
    validate_storage_vector_size(storage, metadata.row_bytes)
}

pub(super) fn validate_row(
    row: &[u8],
    metadata: &PerVectorScalarMetadata,
) -> Result<(), PerVectorScalarError> {
    parse_row(row, metadata).map(|_| ())
}

pub(super) fn parse_row<'a>(
    row: &'a [u8],
    metadata: &PerVectorScalarMetadata,
) -> Result<(PerVectorScalarRowStats, &'a [i8]), PerVectorScalarError> {
    let (stats, codes) = parse_validated_row(row, metadata)?;
    if !stats.scale.is_finite()
        || stats.scale <= 0.0
        || [
            stats.original_norm_upper,
            stats.reconstructed_norm_upper,
            stats.residual_norm_upper,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err(PerVectorScalarError::InvalidRow);
    }
    Ok((stats, codes))
}

pub(super) fn parse_validated_row<'a>(
    row: &'a [u8],
    metadata: &PerVectorScalarMetadata,
) -> Result<(PerVectorScalarRowStats, &'a [i8]), PerVectorScalarError> {
    if row.len() != metadata.row_bytes {
        return Err(PerVectorScalarError::InvalidRow);
    }
    // SAFETY: the row length was validated above. `read_unaligned` supports
    // byte-aligned storage; all persisted words are explicitly little-endian.
    let header = unsafe { std::ptr::read_unaligned(row.as_ptr().cast::<[u32; 4]>()) };
    let scale = f32::from_bits(u32::from_le(header[0]));
    let original_norm_upper = f32::from_bits(u32::from_le(header[1]));
    let reconstructed_norm_upper = f32::from_bits(u32::from_le(header[2]));
    let residual_norm_upper = f32::from_bits(u32::from_le(header[3]));
    let bytes = &row[HEADER_BYTES..];
    // SAFETY: i8 and u8 have identical size/alignment and every bit pattern is
    // valid. The returned slice borrows the same live row.
    let codes = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) };
    Ok((
        PerVectorScalarRowStats {
            scale,
            original_norm_upper,
            reconstructed_norm_upper,
            residual_norm_upper,
        },
        codes,
    ))
}

pub(super) fn to_encoding_error(error: PerVectorScalarError) -> EncodingError {
    EncodingError::ArgumentsError(error.to_string())
}

pub(super) fn invalid_data(error: impl Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}
