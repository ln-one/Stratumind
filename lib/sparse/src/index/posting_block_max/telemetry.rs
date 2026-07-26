use serde::Serialize;

/// Physical counters for the single production PostingBlockMax plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PostingBlockMaxTelemetry {
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
    pub score_buffer_slots: usize,
    pub max_score_buffer_slots: usize,
    pub chunks_decoded: usize,
    pub score_buffer_capacity: usize,
    pub result_buffer_capacity: usize,
    pub active_batch_capacity: usize,
    pub pending_batch_capacity: usize,
    pub pause_count: usize,
}
