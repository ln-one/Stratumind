// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense/Sparse channel execution followed by dynamic WRRF.

mod service;
mod session;
mod source;
mod types;

pub use service::ExactRrfService;
pub use session::ExactHybridSession;
#[cfg(test)]
use source::exact_score_batch_stream;
pub use types::{
    DEFAULT_EXACT_BATCH_SIZE, DEFAULT_SPARSE_POSTING_BATCH_SIZE, ExactRrfRequest, ExactRrfResult,
};

#[cfg(test)]
#[path = "exact_rrf/tests.rs"]
mod tests;
