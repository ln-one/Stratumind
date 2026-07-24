use super::*;

impl PerVectorScalarIndex<QuantizedRamStorage> {
    pub fn build_ram(
        index_directory: &Path,
        vector_storage: &VectorStorageEnum,
        dimension: usize,
        source_generation: u64,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        check_stopped(stopped)?;
        validate_storage(vector_storage, dimension)?;
        fs_err::create_dir_all(index_directory)?;

        let count = vector_storage.total_vector_count();
        let row_bytes = per_vector_scalar_row_bytes(dimension)
            .ok_or_else(|| OperationError::validation_error("PerVectorScalar row size overflow"))?;
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let builder = QuantizedRamStorageBuilder::new(&data_path, count, row_bytes)?;
        build_index(
            vector_storage,
            dimension,
            source_generation,
            &metadata_path,
            builder,
            stopped,
        )
    }

    pub fn load_ram(
        index_directory: &Path,
        expected_source_generation: u64,
    ) -> OperationResult<Self> {
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let metadata = read_metadata(&metadata_path)?;
        validate_source_generation(&metadata, expected_source_generation)?;
        let storage = QuantizedRamStorage::from_file(&data_path, metadata.row_bytes())?;
        load_index(storage, &metadata_path)
    }
}

impl PerVectorScalarIndex<QuantizedStorage<MmapFile>> {
    pub fn build_mmap(
        index_directory: &Path,
        vector_storage: &VectorStorageEnum,
        dimension: usize,
        source_generation: u64,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        check_stopped(stopped)?;
        validate_storage(vector_storage, dimension)?;
        fs_err::create_dir_all(index_directory)?;
        let count = vector_storage.total_vector_count();
        let row_bytes = per_vector_scalar_row_bytes(dimension)
            .ok_or_else(|| OperationError::validation_error("PerVectorScalar row size overflow"))?;
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let builder = QuantizedStorageBuilder::<MmapFile>::new(&data_path, count, row_bytes)?;
        build_index(
            vector_storage,
            dimension,
            source_generation,
            &metadata_path,
            builder,
            stopped,
        )
    }

    pub fn load_mmap(
        index_directory: &Path,
        expected_source_generation: u64,
    ) -> OperationResult<Self> {
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let metadata = read_metadata(&metadata_path)?;
        validate_source_generation(&metadata, expected_source_generation)?;
        let storage =
            QuantizedStorage::<MmapFile>::from_file(&MmapFs, &data_path, metadata.row_bytes())?;
        load_index(storage, &metadata_path)
    }
}

impl PerVectorScalarIndex<QuantizedChunkedMmapStorage> {
    pub fn build_chunked_mmap(
        index_directory: &Path,
        vector_storage: &VectorStorageEnum,
        dimension: usize,
        source_generation: u64,
        in_ram: bool,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        check_stopped(stopped)?;
        validate_storage(vector_storage, dimension)?;
        fs_err::create_dir_all(index_directory)?;
        let row_bytes = per_vector_scalar_row_bytes(dimension)
            .ok_or_else(|| OperationError::validation_error("PerVectorScalar row size overflow"))?;
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let builder = QuantizedChunkedMmapStorageBuilder::new(&data_path, row_bytes, in_ram)?;
        build_index(
            vector_storage,
            dimension,
            source_generation,
            &metadata_path,
            builder,
            stopped,
        )
    }

    pub fn load_chunked_mmap(
        index_directory: &Path,
        expected_source_generation: u64,
        in_ram: bool,
    ) -> OperationResult<Self> {
        let data_path = index_directory.join(DATA_FILE);
        let metadata_path = index_directory.join(METADATA_FILE);
        let metadata = read_metadata(&metadata_path)?;
        validate_source_generation(&metadata, expected_source_generation)?;
        let storage = QuantizedChunkedMmapStorage::new(&data_path, metadata.row_bytes(), in_ram)?;
        load_index(storage, &metadata_path)
    }
}

fn build_index<TStorage, TBuilder>(
    vector_storage: &VectorStorageEnum,
    dimension: usize,
    source_generation: u64,
    metadata_path: &Path,
    builder: TBuilder,
    stopped: &AtomicBool,
) -> OperationResult<PerVectorScalarIndex<TStorage>>
where
    TStorage: EncodedStorage,
    TBuilder: EncodedStorageBuilder<Storage = TStorage>,
{
    let count = vector_storage.total_vector_count();
    let vectors = (0..count).map(|index| {
        match vector_storage.get_vector::<Sequential>(index as PointOffsetType) {
            CowVector::Dense(vector) => vector,
            CowVector::Sparse(_) | CowVector::MultiDense(_) => {
                unreachable!("validated single-vector Dense storage changed type")
            }
        }
    });
    let encoded = EncodedVectorsPerVectorScalar::encode(
        vectors,
        builder,
        &VectorParameters {
            dim: dimension,
            distance_type: distance_type(vector_storage.distance()),
            invert: false,
            deprecated_count: None,
        },
        count,
        source_generation,
        Some(metadata_path),
        stopped,
    )
    .map_err(|error| {
        OperationError::service_error(format!("failed to build PerVectorScalar index: {error}",))
    })?;
    Ok(PerVectorScalarIndex { encoded })
}

fn load_index<TStorage: EncodedStorage>(
    storage: TStorage,
    metadata_path: &Path,
) -> OperationResult<PerVectorScalarIndex<TStorage>> {
    let encoded = EncodedVectorsPerVectorScalar::load(storage, metadata_path)?;
    Ok(PerVectorScalarIndex { encoded })
}

fn read_metadata(metadata_path: &Path) -> OperationResult<PerVectorScalarMetadata> {
    let contents = fs_err::read_to_string(metadata_path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn validate_source_generation(
    metadata: &PerVectorScalarMetadata,
    expected_source_generation: u64,
) -> OperationResult<()> {
    if metadata.source_generation() != expected_source_generation {
        return Err(OperationError::inconsistent_storage(format!(
            "PerVectorScalar source generation mismatch: index={}, Segment={expected_source_generation}",
            metadata.source_generation(),
        )));
    }
    Ok(())
}

pub(super) fn validate_storage(
    vector_storage: &VectorStorageEnum,
    dimension: usize,
) -> OperationResult<()> {
    let expected_layout_bytes = dimension.checked_mul(std::mem::size_of::<f32>());
    let has_expected_dense_layout = expected_layout_bytes.is_some_and(|expected| {
        vector_storage
            .get_vector_layout()
            .is_ok_and(|layout| layout.size() == expected)
    });
    if dimension == 0
        || vector_storage.datatype() != VectorStorageDatatype::Float32
        || !matches!(vector_storage.distance(), Distance::Dot | Distance::Cosine)
        || !has_expected_dense_layout
    {
        return Err(OperationError::validation_error(
            "PerVectorScalar requires matching non-empty Float32 Dot/Cosine single-vector Dense storage",
        ));
    }
    Ok(())
}

fn distance_type(distance: Distance) -> DistanceType {
    match distance {
        Distance::Cosine => DistanceType::Cosine,
        Distance::Dot => DistanceType::Dot,
        Distance::Euclid | Distance::Manhattan => {
            unreachable!("validated Dot/Cosine distance")
        }
    }
}
