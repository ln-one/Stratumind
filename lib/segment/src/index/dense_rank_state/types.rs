// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

const DEFAULT_DENSE_SCALAR_MIN_POINTS: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePhysicalPlan {
    ScalarCertificate,
    PerVectorScalarCertificate,
    ExactScan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseExecutionPolicy {
    pub scalar_min_points: usize,
    pub force_exact_scan: bool,
}

impl Default for DenseExecutionPolicy {
    fn default() -> Self {
        Self {
            scalar_min_points: DEFAULT_DENSE_SCALAR_MIN_POINTS,
            force_exact_scan: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseExecutionTelemetry {
    pub plan: Option<DensePhysicalPlan>,
    pub eligible_points: usize,
    pub accepted_prefix_points: usize,
    pub quantized_scores: usize,
    pub exact_scores: usize,
    pub points_emitted: usize,
}
