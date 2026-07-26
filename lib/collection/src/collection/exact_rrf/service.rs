// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::Duration;

use super::super::Collection;
use super::{ExactHybridSession, ExactRrfRequest, ExactRrfResult};
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::{CollectionError, CollectionResult};

pub(super) struct ExactRrfCancellation {
    stopped: Arc<AtomicBool>,
    armed: bool,
}

impl ExactRrfCancellation {
    pub(super) fn new(stopped: Arc<AtomicBool>) -> Self {
        Self {
            stopped,
            armed: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ExactRrfCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.stopped.store(true, AtomicOrdering::Relaxed);
        }
    }
}

/// Collection boundary for the frozen Production API exact-retrieval plan.
pub struct ExactRrfService<'a> {
    collection: &'a Collection,
}

impl<'a> ExactRrfService<'a> {
    pub fn new(collection: &'a Collection) -> Self {
        Self { collection }
    }

    /// Execute the local exact physical plan when every selected Shard has a
    /// readable local replica. Returns `None` when the safe router must use the
    /// ordinary replica/remote path instead. Plan selection cannot change the
    /// exact result contract.
    pub async fn execute(
        &self,
        request: ExactRrfRequest,
        shard_selection: &ShardSelectorInternal,
        timeout: Option<Duration>,
    ) -> CollectionResult<Option<ExactRrfResult>> {
        let targets = {
            let shard_holder = self.collection.shards_holder.read().await;
            shard_holder
                .select_shards(shard_selection)?
                .into_iter()
                .map(|(shard, _)| Arc::clone(shard))
                .collect::<Vec<_>>()
        };

        let mut snapshots = Vec::with_capacity(targets.len());
        for target in targets {
            let Some(snapshot) = target.exact_segment_read_set().await? else {
                log::debug!("exact RRF skipped: selected Shard has no local snapshot");
                return Ok(None);
            };
            snapshots.push(snapshot);
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let task_stopped = stopped.clone();
        let shard_count = snapshots.len();
        let segment_count = snapshots
            .iter()
            .map(|snapshot| snapshot.segment_count())
            .sum::<usize>();
        // ExactRankSession advances one Segment batch at a time with short
        // Qdrant-runtime tasks. Segment count no longer determines resident
        // reader capacity.
        let required_reader_slots = 1;
        let Some(reservation) = self
            .collection
            .search_runtime
            .try_reserve_exact_session(required_reader_slots)
        else {
            // The ordinary exact path remains available. Crucially, no
            // coordinator or cursor has started, so fallback cannot deadlock
            // behind a partially occupied blocking pool.
            log::debug!(
                "exact RRF skipped: reservation unavailable for {shard_count} Shards, {segment_count} Segments and {required_reader_slots} reader slots",
            );
            return Ok(None);
        };
        debug_assert_eq!(reservation.reader_slots(), required_reader_slots);
        let mut cancellation = ExactRrfCancellation::new(stopped.clone());
        let mut task = reservation
            .coordinator()
            .spawn_blocking(move || {
                let frozen_snapshots = snapshots;
                let pinned_generations = frozen_snapshots
                    .iter()
                    .map(|snapshot| snapshot.pin_generation())
                    .collect::<Vec<_>>();
                let segments = pinned_generations
                    .iter()
                    .map(|(_, segments)| segments.clone())
                    .collect();
                // `frozen_snapshots` stays alive in this closure, so every
                // Shard update guard remains held; `pinned_generations` also
                // keeps optimizer publish/rollback behind the fixed handles.
                let result = ExactHybridSession::new(segments, request, task_stopped).execute();
                drop(pinned_generations);
                drop(frozen_snapshots);
                result
            })
            .expect("fresh exact reservation includes its coordinator slot");
        let result = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, &mut task).await {
                Ok(result) => result,
                Err(_) => {
                    return Err(CollectionError::timeout(timeout, "exact RRF"));
                }
            }
        } else {
            task.await
        }
        .map_err(|error| {
            CollectionError::service_error(format!("exact RRF task failed: {error}"))
        })??;
        cancellation.disarm();

        Ok(Some(ExactRrfResult {
            shard_count,
            ..result
        }))
    }
}
