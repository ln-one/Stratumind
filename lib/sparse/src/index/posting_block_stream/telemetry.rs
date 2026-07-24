use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub enum SparsePhysicalPlan {
    #[default]
    EagerPostingBlock,
    /// Qdrant's original document-at-a-time kernel retained as a resumable
    /// suffix-certified stream.
    NativeSearchContext,
    #[cfg(feature = "stratumind-research")]
    FlatBmp,
    #[cfg(feature = "stratumind-research")]
    SuperblockBmp,
    #[cfg(feature = "stratumind-research")]
    RangeDirectDense,
    #[cfg(feature = "stratumind-research")]
    RangeDirectTouched,
    #[cfg(feature = "stratumind-research")]
    RangeDirectSorted,
    PostingBlockMax,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseExecutionPlan {
    Auto,
    EagerPostingBlock,
    NativeSearchContext,
    #[cfg(feature = "stratumind-research")]
    FlatBmp,
    #[cfg(feature = "stratumind-research")]
    SuperblockBmp,
    PostingBlockMax,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostingBlockStreamTelemetry {
    pub plan: SparsePhysicalPlan,
    pub cursor_started: bool,
    pub query_terms: usize,
    pub query_posting_elements: usize,
    pub persisted_query_terms: usize,
    pub posting_lists: usize,
    pub batches: usize,
    pub bound_evaluations: usize,
    pub zero_bound_batches: usize,
    pub batches_expanded: usize,
    pub posting_elements_visited: usize,
    pub nonzero_documents_scored: usize,
    pub points_emitted: usize,
    pub max_pending_batches: usize,
    pub max_pending_points: usize,
    pub max_buffered_points: usize,
    pub hierarchy_nodes_expanded: usize,
    pub sidecar_bytes: usize,
    pub range_direct_batches: usize,
    pub score_buffer_slots: usize,
    pub score_buffer_touched: usize,
    pub max_score_buffer_slots: usize,
    pub max_touched_slots: usize,
    pub sorted_batches: usize,
    pub chunks_decoded: usize,
    pub score_buffer_capacity: usize,
    pub result_buffer_capacity: usize,
    pub active_batch_capacity: usize,
    pub pending_batch_capacity: usize,
    pub pause_count: usize,
    #[cfg(feature = "stratumind-research")]
    pub phase: PostingBlockPhaseTelemetry,
}

#[cfg(feature = "stratumind-research")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostingBlockPhaseTelemetry {
    pub posting_open_ns: u64,
    pub bound_plan_ns: u64,
    pub bound_heapify_ns: u64,
    pub range_score_ns: u64,
    pub score_scan_ns: u64,
    pub result_heapify_ns: u64,
    pub proof_ns: u64,
    pub delivery_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub enum PostingBlockMaxVariant {
    #[default]
    V1,
    #[cfg(feature = "stratumind-research")]
    RangeDirectDense,
    #[cfg(feature = "stratumind-research")]
    RangeDirectTouched,
    #[cfg(feature = "stratumind-research")]
    RangeDirectSorted,
    CompressedMetadata,
}

impl PostingBlockMaxVariant {
    pub(super) fn physical_plan(self) -> SparsePhysicalPlan {
        match self {
            Self::V1 => SparsePhysicalPlan::EagerPostingBlock,
            #[cfg(feature = "stratumind-research")]
            Self::RangeDirectDense => SparsePhysicalPlan::RangeDirectDense,
            #[cfg(feature = "stratumind-research")]
            Self::RangeDirectTouched => SparsePhysicalPlan::RangeDirectTouched,
            #[cfg(feature = "stratumind-research")]
            Self::RangeDirectSorted => SparsePhysicalPlan::RangeDirectSorted,
            Self::CompressedMetadata => SparsePhysicalPlan::PostingBlockMax,
        }
    }
}
