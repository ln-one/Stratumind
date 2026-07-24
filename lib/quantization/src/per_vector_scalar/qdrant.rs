use super::format::parse_validated_row;
use super::*;

impl<TStorage: EncodedStorage> EncodedVectors for EncodedVectorsPerVectorScalar<TStorage> {
    type EncodedQuery = EncodedQueryPerVectorScalar;

    fn is_in_ram_or_mmap() -> bool {
        TStorage::is_in_ram_or_mmap()
    }

    fn is_on_disk(&self) -> bool {
        self.encoded_vectors.is_on_disk()
    }

    fn encode_query(&self, query: &[f32]) -> Self::EncodedQuery {
        self.try_encode_query(query)
            .expect("Qdrant must validate a Dense query before quantized scoring")
    }

    fn iter_batch(
        &self,
        offsets: &[PointOffsetType],
    ) -> impl Iterator<Item = (usize, Cow<'_, [u8]>)> {
        self.encoded_vectors.iter_batch(offsets)
    }

    fn score(
        &self,
        query: &Self::EncodedQuery,
        encoded_vector: &[u8],
        hardware_counter: &HardwareCounterCell,
    ) -> f32 {
        self.score_bytes(True, query, encoded_vector, hardware_counter)
    }

    fn score_point(
        &self,
        query: &Self::EncodedQuery,
        i: PointOffsetType,
        hardware_counter: &HardwareCounterCell,
    ) -> f32 {
        hardware_counter
            .cpu_counter()
            .incr_delta(self.metadata.dimension);
        let row = self.encoded_vectors.get_vector_data(i);
        self.center_point(query, &row)
            .map(|center| center as f32)
            .unwrap_or(f32::NAN)
    }

    fn score_internal(
        &self,
        i: PointOffsetType,
        j: PointOffsetType,
        hardware_counter: &HardwareCounterCell,
    ) -> f32 {
        hardware_counter
            .vector_io_read()
            .incr_delta(self.metadata.row_bytes.saturating_mul(2));
        let Some(query) = self.encode_internal_vector(i) else {
            return f32::NAN;
        };
        self.score_point(&query, j, hardware_counter)
    }

    fn quantized_vector_size(&self) -> usize {
        self.metadata.row_bytes
    }

    fn encode_internal_vector(&self, id: PointOffsetType) -> Option<Self::EncodedQuery> {
        let row = self.encoded_vectors.get_vector_data(id);
        let (stats, codes) = parse_validated_row(&row, &self.metadata).ok()?;
        Some(EncodedQueryPerVectorScalar {
            scale: stats.scale,
            codes: codes.to_vec(),
            original_norm_upper: f64::from(stats.original_norm_upper),
            reconstructed_norm_upper: f64::from(stats.reconstructed_norm_upper),
            residual_norm_upper: f64::from(stats.residual_norm_upper),
        })
    }

    fn upsert_vector(
        &mut self,
        _id: PointOffsetType,
        _vector: &[f32],
        _hardware_counter: &HardwareCounterCell,
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "PerVectorScalar immutable storage does not support upsert",
        ))
    }

    fn vectors_count(&self) -> usize {
        self.encoded_vectors.vectors_count()
    }

    fn flusher(&self) -> MmapFlusher {
        self.encoded_vectors.flusher()
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut files = self.encoded_vectors.files();
        if let Some(metadata_path) = &self.metadata_path {
            files.push(metadata_path.clone());
        }
        files
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        let mut files = self.encoded_vectors.immutable_files();
        if let Some(metadata_path) = &self.metadata_path {
            files.push(metadata_path.clone());
        }
        files
    }

    fn heap_size_bytes(&self) -> usize {
        self.encoded_vectors.heap_size_bytes()
    }

    type SupportsBytes = True;

    fn score_bytes(
        &self,
        _: Self::SupportsBytes,
        query: &Self::EncodedQuery,
        bytes: &[u8],
        hardware_counter: &HardwareCounterCell,
    ) -> f32 {
        hardware_counter
            .cpu_counter()
            .incr_delta(self.metadata.dimension);
        self.center(query, bytes)
            .map(|center| center as f32)
            .unwrap_or(f32::NAN)
    }
}
