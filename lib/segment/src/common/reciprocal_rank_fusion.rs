// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Reciprocal Rank Fusion (RRF) combines rankings from multiple sources.
//! See <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>.
//!
//! Stratumind extends RRF with deterministic composition of recoverable,
//! exact rank streams and a certificate that stops once the fused Top-K is
//! immutable.

mod executor;
mod scoring;
mod session;
mod state;
mod types;

pub use executor::{
    execute_dynamic_rrf, execute_dynamic_rrf_batches_with_policy, execute_dynamic_rrf_with_policy,
};
pub use scoring::{exact_rrf_scoring, rrf_scoring};
use scoring::{guaranteed_before, position_score, rrf_order};
pub use session::DynamicRrfSession;
pub use state::DynamicRrfState;
pub use types::{
    ChannelRankBatch, DynamicRrfAdvance, DynamicRrfExecution, DynamicRrfPolicy,
    DynamicRrfScheduler, DynamicRrfStopReason, ExactRrfBatchStream, ExactRrfStream,
    infallible_exact_rrf_stream,
};

/// Default rank constant used by Qdrant's RRF implementation.
pub const DEFAULT_RRF_K: usize = 2;

#[cfg(test)]
mod tests;
