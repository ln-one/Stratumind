// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Exact Dense ranking state and physical-plan policy.

mod core;
mod types;

pub use core::DenseRankState;
pub(crate) use core::PendingBound;

pub use types::{DenseExecutionPolicy, DenseExecutionTelemetry, DensePhysicalPlan};
