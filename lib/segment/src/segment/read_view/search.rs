use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;

use ahash::AHashMap;
use common::counter::hardware_counter::HardwareCounterCell;
use common::iterator_ext::IteratorExt;
use common::types::{DeferredBehavior, ScoredPointOffset};
use sparse::SearchScratchArena;
use sparse::common::sparse_vector::SparseVector;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::{check_query_vectors, check_stopped};
use crate::data_types::modifier::Modifier;
use crate::data_types::query_context::{QueryContext, QueryIdfStats, SegmentQueryContext};
use crate::data_types::segment_record::{NamedVectorsOwned, SegmentRecord};
use crate::data_types::vectors::{QueryVector, VectorStructInternal};
use crate::id_tracker::IdTrackerRead;
use crate::index::dense_rank_state::{DenseExecutionPolicy, DenseRankState};
use crate::index::sparse_rank_state::ExactSparseIndexState;
use crate::index::{PayloadIndexRead, VectorIndexRead};
use crate::payload_storage::PayloadStorageRead;
use crate::segment::read_view::{SegmentReadView, SegmentReadViewFor};
use crate::segment::vector_data_read::VectorDataRead;
use crate::types::{
    ExtendedPointId, Filter, PointIdType, ScoredPoint, SearchParams, VectorName, VectorNameBuf,
    WithPayload, WithVector,
};
use crate::vector_storage::raw_scorer::new_raw_scorer;
use crate::vector_storage::{VectorStorageRead, check_deleted_condition};

pub struct SparseRankState {
    index: ExactSparseIndexState,
    internal_buffer: VecDeque<ScoredPointOffset>,
    external_buffer: VecDeque<ScoredPoint>,
    lookahead: Option<ScoredPoint>,
    posting_batch_size: usize,
}

pub struct DenseSegmentRankState {
    rank: DenseRankState,
    internal_buffer: VecDeque<ScoredPointOffset>,
    external_buffer: VecDeque<ScoredPoint>,
    lookahead: Option<ScoredPoint>,
}

impl SegmentReadViewFor<'_> {
    /// Build reader-independent Dense rank state. The returned state contains
    /// no Segment borrow and can be moved between short-lived search tasks.
    pub fn open_exact_dense_rank_state(
        &self,
        vector_name: &VectorName,
        query: &[f32],
        filter: Option<&Filter>,
        policy: DenseExecutionPolicy,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<DenseRankState> {
        if query.is_empty() || query.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(OperationError::validation_error(
                "exact Dense session requires a non-empty finite Query",
            ));
        }
        let vector_data = self
            .vector_data
            .get(vector_name)
            .ok_or_else(|| OperationError::vector_name_not_exists(vector_name))?;
        let vector_query_context = query_context.get_vector_context(vector_name);
        let hardware_counter = vector_query_context.hardware_counter();
        let stopped = vector_query_context.is_stopped();
        let vector_index = vector_data.vector_index();
        let quantized = vector_index.quantized_vectors();
        let quantized = quantized.as_ref().map(|quantized| quantized.borrow());
        let quantized = quantized.as_ref().and_then(|quantized| quantized.as_ref());
        let vector_storage = vector_data.vector_storage();
        let deleted_vectors = vector_storage.deleted_vector_bitslice();
        let deleted_points = self.id_tracker.deleted_point_bitslice();
        let deferred_from = self.id_tracker.deferred_internal_id();
        let filter_context = filter
            .map(|filter| self.payload_index.filter_context(filter, &hardware_counter))
            .transpose()?;
        let eligible = (0..vector_storage.total_vector_count() as u32)
            .filter(|&point| {
                check_deleted_condition(point, deleted_vectors, deleted_points)
                    && deferred_from.is_none_or(|deferred| point < deferred)
                    && filter_context
                        .as_ref()
                        .is_none_or(|context| context.check(point))
            })
            .collect();
        DenseRankState::new(
            &vector_storage,
            quantized,
            eligible,
            query,
            policy,
            &hardware_counter,
            &stopped,
        )
    }

    /// Advance owned Dense state while borrowing this Segment only for the
    /// exact-score batch.
    pub fn advance_exact_dense_rank_state(
        &self,
        vector_name: &VectorName,
        query: &[f32],
        state: &mut DenseRankState,
        max_results: usize,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        let vector_data = self
            .vector_data
            .get(vector_name)
            .ok_or_else(|| OperationError::vector_name_not_exists(vector_name))?;
        let vector_query_context = query_context.get_vector_context(vector_name);
        let hardware_counter = vector_query_context.hardware_counter();
        let vector_storage = vector_data.vector_storage();
        let query_vector: QueryVector =
            crate::data_types::vectors::VectorInternal::Dense(query.to_vec()).into();
        let scorer = new_raw_scorer(query_vector, &vector_storage, hardware_counter.fork())?;
        state.next_batch_with(max_results, |ids, scores| {
            scorer.score_points(ids, scores);
        })
    }

    pub fn open_exact_dense_segment_state(
        &self,
        vector_name: &VectorName,
        query: &[f32],
        filter: Option<&Filter>,
        policy: DenseExecutionPolicy,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<DenseSegmentRankState> {
        Ok(DenseSegmentRankState {
            rank: self.open_exact_dense_rank_state(
                vector_name,
                query,
                filter,
                policy,
                query_context,
            )?,
            internal_buffer: VecDeque::new(),
            external_buffer: VecDeque::new(),
            lookahead: None,
        })
    }

    pub fn advance_exact_dense_segment_state(
        &self,
        vector_name: &VectorName,
        query: &[f32],
        state: &mut DenseSegmentRankState,
        max_results: usize,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Vec<ScoredPoint>> {
        if max_results == 0 {
            return Err(OperationError::validation_error(
                "exact Dense delivery batch must be positive",
            ));
        }
        let mut pull_visible = || -> OperationResult<Option<ScoredPoint>> {
            if state.internal_buffer.is_empty() {
                state.internal_buffer = VecDeque::from(self.advance_exact_dense_rank_state(
                    vector_name,
                    query,
                    &mut state.rank,
                    max_results.max(16),
                    query_context,
                )?);
                if state.internal_buffer.is_empty() {
                    return Ok(None);
                }
            }
            let point = state
                .internal_buffer
                .pop_front()
                .expect("non-empty Dense internal batch");
            let id = self.id_tracker.external_id(point.idx).ok_or_else(|| {
                OperationError::inconsistent_storage(format!(
                    "exact Dense rank state returned unmapped internal point {}",
                    point.idx
                ))
            })?;
            let version = self.id_tracker.internal_version(point.idx).ok_or_else(|| {
                OperationError::inconsistent_storage(format!(
                    "exact Dense rank state returned unversioned point {id}"
                ))
            })?;
            Ok(Some(ScoredPoint {
                id,
                version,
                score: point.score,
                payload: None,
                vector: None,
                shard_key: None,
                order_value: None,
            }))
        };

        let mut output = Vec::with_capacity(max_results);
        while output.len() < max_results {
            if state.external_buffer.is_empty() {
                let Some(first) = state
                    .lookahead
                    .take()
                    .map_or_else(&mut pull_visible, |point| Ok(Some(point)))?
                else {
                    break;
                };
                let score = first.score;
                let mut group = vec![first];
                loop {
                    match pull_visible()? {
                        Some(point) if point.score == score => group.push(point),
                        Some(point) => {
                            state.lookahead = Some(point);
                            break;
                        }
                        None => break,
                    }
                }
                group.sort_unstable_by(|left, right| left.id.cmp(&right.id));
                state.external_buffer = VecDeque::from(group);
            }
            let count = (max_results - output.len()).min(state.external_buffer.len());
            output.extend(state.external_buffer.drain(..count));
        }
        Ok(output)
    }

    pub fn open_exact_sparse_rank_state(
        &self,
        vector_name: &VectorName,
        query: &SparseVector,
        posting_batch_size: usize,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<SparseRankState> {
        if posting_batch_size == 0 {
            return Err(OperationError::validation_error(
                "exact Sparse posting batch size must be positive",
            ));
        }
        let vector_data = self
            .vector_data
            .get(vector_name)
            .ok_or_else(|| OperationError::vector_name_not_exists(vector_name))?;
        let vector_query_context = query_context.get_vector_context(vector_name);
        if self
            .segment_config
            .sparse_vector_data
            .get(vector_name)
            .is_some_and(|config| config.modifier == Some(Modifier::Idf))
            || vector_query_context.is_require_idf()
        {
            return Err(OperationError::WrongSparse);
        }
        let arena = SearchScratchArena::new_slow();
        let vector_index = vector_data.vector_index();
        let index = ExactSparseIndexState::open(
            &vector_index,
            query,
            posting_batch_size,
            &arena,
            &vector_query_context.hardware_counter(),
        )?;
        Ok(SparseRankState {
            index,
            internal_buffer: VecDeque::new(),
            external_buffer: VecDeque::new(),
            lookahead: None,
            posting_batch_size,
        })
    }

    pub fn advance_exact_sparse_rank_state(
        &self,
        vector_name: &VectorName,
        filter: Option<&Filter>,
        state: &mut SparseRankState,
        max_results: usize,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Vec<ScoredPoint>> {
        if max_results == 0 {
            return Err(OperationError::validation_error(
                "exact Sparse delivery batch must be positive",
            ));
        }
        let vector_data = self
            .vector_data
            .get(vector_name)
            .ok_or_else(|| OperationError::vector_name_not_exists(vector_name))?;
        let vector_query_context = query_context.get_vector_context(vector_name);
        let hardware_counter = vector_query_context.hardware_counter();
        let stopped = vector_query_context.is_stopped();
        let vector_index = vector_data.vector_index();
        let vector_storage = vector_data.vector_storage();
        let deleted_vectors = vector_storage.deleted_vector_bitslice();
        let deleted_points = self.id_tracker.deleted_point_bitslice();
        let deferred_from = self.id_tracker.deferred_internal_id();
        let filter_context = filter
            .map(|filter| self.payload_index.filter_context(filter, &hardware_counter))
            .transpose()?;
        let arena = SearchScratchArena::new_slow();

        let mut pull_visible = || -> OperationResult<Option<ScoredPoint>> {
            loop {
                check_stopped(&stopped)?;
                if state.internal_buffer.is_empty() {
                    state.internal_buffer = VecDeque::from(state.index.next_batch(
                        &vector_index,
                        state.posting_batch_size,
                        &arena,
                        &hardware_counter,
                        &stopped,
                    )?);
                    if state.internal_buffer.is_empty() {
                        return Ok(None);
                    }
                }
                let point = state
                    .internal_buffer
                    .pop_front()
                    .expect("non-empty Sparse internal batch");
                if !check_deleted_condition(point.idx, deleted_vectors, deleted_points)
                    || deferred_from.is_some_and(|deferred| point.idx >= deferred)
                    || filter_context
                        .as_ref()
                        .is_some_and(|context| !context.check(point.idx))
                {
                    continue;
                }
                let id = self.id_tracker.external_id(point.idx).ok_or_else(|| {
                    OperationError::inconsistent_storage(format!(
                        "exact Sparse rank state returned unmapped internal point {}",
                        point.idx
                    ))
                })?;
                let version = self.id_tracker.internal_version(point.idx).ok_or_else(|| {
                    OperationError::inconsistent_storage(format!(
                        "exact Sparse rank state returned unversioned point {id}"
                    ))
                })?;
                return Ok(Some(ScoredPoint {
                    id,
                    version,
                    score: point.score,
                    payload: None,
                    vector: None,
                    shard_key: None,
                    order_value: None,
                }));
            }
        };

        let mut output = Vec::with_capacity(max_results);
        while output.len() < max_results {
            if state.external_buffer.is_empty() {
                let Some(first) = state
                    .lookahead
                    .take()
                    .map_or_else(&mut pull_visible, |point| Ok(Some(point)))?
                else {
                    break;
                };
                let score = first.score;
                let mut group = vec![first];
                loop {
                    match pull_visible()? {
                        Some(point) if point.score == score => group.push(point),
                        Some(point) => {
                            state.lookahead = Some(point);
                            break;
                        }
                        None => break,
                    }
                }
                group.sort_unstable_by(|left, right| left.id.cmp(&right.id));
                state.external_buffer = VecDeque::from(group);
            }
            let count = (max_results - output.len()).min(state.external_buffer.len());
            output.extend(state.external_buffer.drain(..count));
        }
        Ok(output)
    }
}

impl<'s, TIdT, TPI, TPS, TVD> SegmentReadView<'s, TIdT, TPI, TPS, TVD>
where
    TIdT: IdTrackerRead,
    TPI: PayloadIndexRead,
    TPS: PayloadStorageRead,
    TVD: VectorDataRead,
{
    /// Reads records from the segment for the given external point IDs,
    /// optionally enriched with vectors and payload.
    pub fn retrieve(
        &self,
        point_ids: &[PointIdType],
        with_payload: &WithPayload,
        with_vector: &WithVector,
        hw_counter: &HardwareCounterCell,
        is_stopped: &AtomicBool,
        deferred_behavior: DeferredBehavior,
    ) -> OperationResult<AHashMap<ExtendedPointId, SegmentRecord>> {
        // Stage 1: resolve external → internal once, into two parallel vectors.
        // The id tracker owns this: deferred filtering happens inline (no
        // `point_is_deferred` lookup). The parallel-vector shape lets
        // a future batched payload / vector fetcher consume `&offsets`
        // straight without unzipping first.
        let (resolved_ids, resolved_offsets) = self
            .id_tracker
            .resolve_external_ids(point_ids, deferred_behavior);
        debug_assert_eq!(resolved_ids.len(), resolved_offsets.len());

        // Stage 2: pre-allocate one record per resolved point. The `vectors`
        // slot is initialised here according to `with_vector`, so the
        // `WithVector::Bool(false)` path needs no separate clearing pass.
        let needs_vectors = match with_vector {
            WithVector::Bool(true) | WithVector::Selector(_) => true,
            WithVector::Bool(false) => false,
        };
        let mut records: AHashMap<ExtendedPointId, SegmentRecord> = resolved_ids
            .iter()
            .map(|&id| {
                let record = SegmentRecord {
                    id,
                    vectors: needs_vectors.then(NamedVectorsOwned::default),
                    payload: None,
                };
                (id, record)
            })
            .collect();

        // Stage 3: vectors. The external id rides along as the read's user
        // data — it comes back unchanged in the callback, so no extra
        // `offset → id` lookup is needed.
        if needs_vectors {
            let mut process_vectors = |vector_name: &VectorNameBuf| -> OperationResult<()> {
                let keys = resolved_ids
                    .iter()
                    .zip(&resolved_offsets)
                    .map(|(&id, &offset)| (id, offset))
                    .stop_if(is_stopped);
                self.vectors_by_offsets(vector_name, keys, hw_counter, |id, _offset, vec| {
                    if let Some(record) = records.get_mut(&id) {
                        record
                            .vectors
                            .as_mut()
                            .expect("needs_vectors path keeps vectors as Some")
                            .push((vector_name.clone(), vec));
                    }
                })
            };

            match with_vector {
                WithVector::Bool(true) => {
                    for vector_name in self.vector_data.keys() {
                        process_vectors(vector_name)?;
                    }
                }
                WithVector::Selector(names) => {
                    for vector_name in names {
                        process_vectors(vector_name)?;
                    }
                }
                WithVector::Bool(false) => unreachable!("guarded by needs_vectors"),
            }
        }

        // Stage 4: payload. Use the already-resolved offsets to skip another
        // external→internal lookup per point. The per-iteration shape here
        // mirrors what a future batched payload fetcher would consume —
        // `&resolved_offsets` becomes its input directly.
        if with_payload.enable {
            for (&id, &offset) in resolved_ids.iter().zip(&resolved_offsets) {
                check_stopped(is_stopped)?;
                let payload = self.payload_by_offset(offset, hw_counter)?;
                let payload = match &with_payload.payload_selector {
                    Some(selector) => selector.process(payload),
                    None => payload,
                };
                if let Some(record) = records.get_mut(&id) {
                    record.payload = Some(payload);
                }
            }
        }

        Ok(records)
    }

    /// Converts raw `ScoredPointOffset` search results into user-facing
    /// `ScoredPoint`s. Deferred points are filtered out.
    pub fn process_search_result(
        &self,
        internal_result: Vec<ScoredPointOffset>,
        with_payload: &WithPayload,
        with_vector: &WithVector,
        hw_counter: &HardwareCounterCell,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPoint>> {
        let (point_ids, scored_offsets): (Vec<_>, Vec<_>) = internal_result
            .into_iter()
            .filter_map(|scored_point_offset| {
                let point_offset = scored_point_offset.idx;
                let point_id = self.id_tracker.external_id(point_offset);
                // This can happen if a point was modified between retrieving and post-processing,
                // but this function locks the segment so it can't be modified during execution.
                debug_assert!(
                    point_id.is_some(),
                    "Point with internal ID {point_offset} not found in id tracker"
                );
                point_id.map(|id| (id, scored_point_offset))
            })
            .unzip();

        let mut segment_records = self.retrieve(
            &point_ids,
            with_payload,
            with_vector,
            hw_counter,
            is_stopped,
            DeferredBehavior::Exclude,
        )?;

        let mut results = Vec::with_capacity(point_ids.len());

        for (point_id, scored_offset) in point_ids.into_iter().zip(scored_offsets) {
            let ScoredPointOffset {
                idx: point_offset,
                score: point_score,
            } = scored_offset;

            let record = segment_records.remove(&point_id);

            // It is still possible for scored points to have duplicates for some reason, so we
            // probably don't want to return error in release mode. We also don't want to copy all
            // data just to handle this unexpected case.
            let Some(record) = record else {
                debug_assert!(
                    false,
                    "Record for point ID {point_id} not found during search result processing"
                );
                continue;
            };

            let point_version =
                self.id_tracker
                    .internal_version(point_offset)
                    .ok_or_else(|| {
                        OperationError::service_error(format!(
                            "Corrupter id_tracker, no version for point {point_id}"
                        ))
                    })?;

            let SegmentRecord {
                id,
                vectors,
                payload,
            } = record;

            results.push(ScoredPoint {
                id,
                version: point_version,
                score: point_score,
                payload,
                vector: vectors.map(VectorStructInternal::from),
                shard_key: None,
                order_value: None,
            });
        }

        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search_batch(
        &self,
        vector_name: &VectorName,
        query_vectors: &[&QueryVector],
        with_payload: &WithPayload,
        with_vector: &WithVector,
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPoint>>> {
        check_query_vectors(vector_name, query_vectors, self.segment_config)?;
        let vector_data = self
            .vector_data
            .get(vector_name)
            .ok_or_else(|| OperationError::vector_name_not_exists(vector_name))?;
        let vector_query_context = query_context.get_vector_context(vector_name);
        let internal_results = vector_data.vector_index().search(
            query_vectors,
            filter,
            top,
            params,
            &vector_query_context,
        )?;

        check_stopped(&vector_query_context.is_stopped())?;

        let hw_counter = vector_query_context.hardware_counter();

        internal_results
            .into_iter()
            .map(|internal_result| {
                self.process_search_result(
                    internal_result,
                    with_payload,
                    with_vector,
                    &hw_counter,
                    &vector_query_context.is_stopped(),
                )
            })
            .collect()
    }

    pub fn fill_query_context(&self, query_context: &mut QueryContext) -> OperationResult<()> {
        query_context.add_available_point_count(self.available_point_count_without_deferred());
        let hw_acc = query_context.hardware_usage_accumulator();
        let hw_counter = hw_acc.get_counter_cell();

        let QueryIdfStats {
            idf,
            indexed_vectors,
        } = query_context.mut_idf_stats();

        for (vector_name, idf) in idf.iter_mut() {
            if let Some(vector_data) = self.vector_data.get(vector_name) {
                let vector_index = vector_data.vector_index();

                let indexed_vector_count = vector_index.indexed_vector_count();

                if let Some(count) = indexed_vectors.get_mut(vector_name) {
                    *count += indexed_vector_count;
                } else {
                    indexed_vectors.insert(vector_name.clone(), indexed_vector_count);
                }

                vector_index.fill_idf_statistics(idf, &hw_counter)?;
            }
        }
        Ok(())
    }
}
