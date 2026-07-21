// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Identity and telemetry adapters for exact channel rank streams.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

use sparse::index::adaptive_top_k_stream::AdaptiveTopKStream;
use sparse::index::block_max::BlockMaxStream;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::posting_block_stream::{NativeCertifiedSparseCursor, NativeSparseCursorError};
use sparse::index::shared_posting_block_stream::SharedPostingBlockStream;

use super::dense_quantized::DenseQuantizedStream;
use super::n_channel_exact::{ExactChannelTelemetry, SharedSparseChannelTelemetry};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::reciprocal_rank_fusion::ExactRrfStream;
use crate::types::ExtendedPointId;

pub(super) struct CancellableIdentityStream<'a> {
    pub(super) inner: ExactRrfStream<'a>,
    pub(super) stopped: &'a AtomicBool,
}

impl Iterator for CancellableIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped.load(Relaxed) {
            return Some(Err(OperationError::cancelled(
                "exact rank stream cancelled before the next pull",
            )));
        }
        self.inner.next()
    }
}

pub(super) struct SharedPostingIdentityStream<'a> {
    pub(super) inner: SharedPostingBlockStream<'a>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
}

impl Iterator for SharedPostingIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry
            .set(ExactChannelTelemetry::SparseSharedPosting(
                self.inner.telemetry(),
            ));
        point.map(|point| Ok(ExtendedPointId::from(u64::from(point.idx))))
    }
}

pub(super) struct SharedSparseIdentityStream {
    pub(super) inner: std::vec::IntoIter<common::types::ScoredPointOffset>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
}

impl Iterator for SharedSparseIdentityStream {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        if point.is_some() {
            let ExactChannelTelemetry::SparseShared(current) = self.telemetry.get() else {
                unreachable!("Shared Sparse stream telemetry variant")
            };
            self.telemetry.set(ExactChannelTelemetry::SparseShared(
                SharedSparseChannelTelemetry {
                    points_emitted: current.points_emitted + 1,
                },
            ));
        }
        point.map(|point| Ok(ExtendedPointId::from(u64::from(point.idx))))
    }
}

pub(super) struct DenseIdentityStream<'a> {
    pub(super) inner: DenseQuantizedStream<'a>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
}

impl Iterator for DenseIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry
            .set(ExactChannelTelemetry::Dense(self.inner.telemetry()));
        point.map(|point| Ok(ExtendedPointId::from(u64::from(point.id))))
    }
}

pub(super) struct SparseIdentityStream<'a> {
    pub(super) inner: BlockMaxStream<'a>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
}

impl Iterator for SparseIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry
            .set(ExactChannelTelemetry::Sparse(self.inner.telemetry()));
        point.map(|point| Ok(ExtendedPointId::from(u64::from(point.idx))))
    }
}

pub(super) struct PostingSparseIdentityStream<'a> {
    pub(super) inner: NativeCertifiedSparseCursor<'a, InvertedIndexCompressedImmutableRam<f32>>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
    pub(super) stopped: &'a AtomicBool,
}

impl Iterator for PostingSparseIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next_result(self.stopped);
        self.telemetry
            .set(ExactChannelTelemetry::SparsePosting(self.inner.telemetry()));
        match point {
            Ok(Some(point)) => Some(Ok(ExtendedPointId::from(u64::from(point.idx)))),
            Ok(None) => None,
            Err(NativeSparseCursorError::Cancelled) => Some(Err(OperationError::cancelled(
                "native Sparse cursor cancelled before exact EOF",
            ))),
            Err(error) => Some(Err(OperationError::service_error_light(error.to_string()))),
        }
    }
}

pub(super) struct AdaptiveSparseIdentityStream<'a> {
    pub(super) inner: AdaptiveTopKStream<'a, InvertedIndexCompressedImmutableRam<f32>>,
    pub(super) telemetry: Rc<Cell<ExactChannelTelemetry>>,
}

impl Iterator for AdaptiveSparseIdentityStream<'_> {
    type Item = OperationResult<ExtendedPointId>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.inner.next();
        self.telemetry.set(ExactChannelTelemetry::SparseAdaptive(
            self.inner.telemetry(),
        ));
        point.map(|point| Ok(ExtendedPointId::from(u64::from(point.idx))))
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn cancellation_stops_identity_stream_before_the_next_pull() {
        let stopped = AtomicBool::new(false);
        let point = ExtendedPointId::from(7_u64);
        let inner = Box::new(std::iter::once(Ok(point))) as ExactRrfStream<'_>;
        let mut stream = CancellableIdentityStream {
            inner,
            stopped: &stopped,
        };

        stopped.store(true, Relaxed);

        assert!(matches!(
            stream.next(),
            Some(Err(OperationError::Cancelled { .. }))
        ));
    }
}
