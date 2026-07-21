// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Physical sharing beneath logically independent exact rank streams.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::common::reciprocal_rank_fusion::ExactRrfStream;
use crate::types::ExtendedPointId;

struct Coordinator<'a> {
    inner: Option<ExactRrfStream<'a>>,
    buffer: VecDeque<ExtendedPointId>,
    memory: Option<ReplayBufferTracker>,
    base_position: usize,
    reader_positions: Vec<Option<usize>>,
    exhausted: bool,
    cancelled: bool,
    physical_pulls: usize,
    peak_buffered_identities: usize,
}

struct Reader<'a> {
    coordinator: Rc<RefCell<Coordinator<'a>>>,
    reader: usize,
}

#[derive(Clone)]
pub struct SharedRankStream<'a> {
    coordinator: Rc<RefCell<Coordinator<'a>>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedRankStreamTelemetry {
    pub active_readers: usize,
    pub physical_pulls: usize,
    pub buffered_identities: usize,
    pub peak_buffered_identities: usize,
    pub exhausted: bool,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ReplayBufferTracker {
    state: Rc<RefCell<ReplayBufferMemory>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ReplayBufferMemory {
    current_identities: usize,
    peak_identities: usize,
}

impl ReplayBufferTracker {
    fn grow(&self, identities: usize) {
        let mut state = self.state.borrow_mut();
        state.current_identities += identities;
        state.peak_identities = state.peak_identities.max(state.current_identities);
    }

    fn shrink(&self, identities: usize) {
        let mut state = self.state.borrow_mut();
        state.current_identities = state
            .current_identities
            .checked_sub(identities)
            .expect("replay memory accounting cannot underflow");
    }

    pub(crate) fn peak_identities(&self) -> usize {
        self.state.borrow().peak_identities
    }
}

impl<'a> SharedRankStream<'a> {
    pub fn new(inner: ExactRrfStream<'a>) -> Self {
        Self::new_inner(inner, None)
    }

    pub(crate) fn new_tracked(inner: ExactRrfStream<'a>, memory: ReplayBufferTracker) -> Self {
        Self::new_inner(inner, Some(memory))
    }

    fn new_inner(inner: ExactRrfStream<'a>, memory: Option<ReplayBufferTracker>) -> Self {
        Self {
            coordinator: Rc::new(RefCell::new(Coordinator {
                inner: Some(inner),
                buffer: VecDeque::new(),
                memory,
                base_position: 0,
                reader_positions: Vec::new(),
                exhausted: false,
                cancelled: false,
                physical_pulls: 0,
                peak_buffered_identities: 0,
            })),
        }
    }

    /// Adds a reader at rank zero. All DAG edges are registered before the
    /// first root pull, so late subscriptions after production starts are
    /// rejected rather than silently returning a suffix.
    pub fn subscribe(&self) -> ExactRrfStream<'a> {
        let mut coordinator = self.coordinator.borrow_mut();
        assert_eq!(
            coordinator.physical_pulls, 0,
            "a shared exact rank stream cannot add a rank-zero reader after production starts",
        );
        let reader = coordinator.reader_positions.len();
        coordinator.reader_positions.push(Some(0));
        drop(coordinator);
        Box::new(Reader {
            coordinator: Rc::clone(&self.coordinator),
            reader,
        })
    }

    pub fn telemetry(&self) -> SharedRankStreamTelemetry {
        let coordinator = self.coordinator.borrow();
        SharedRankStreamTelemetry {
            active_readers: coordinator.reader_positions.iter().flatten().count(),
            physical_pulls: coordinator.physical_pulls,
            buffered_identities: coordinator.buffer.len(),
            peak_buffered_identities: coordinator.peak_buffered_identities,
            exhausted: coordinator.exhausted,
            cancelled: coordinator.cancelled,
        }
    }
}

impl Coordinator<'_> {
    fn prune_consumed(&mut self) {
        let retained_from = self
            .reader_positions
            .iter()
            .flatten()
            .copied()
            .min()
            .unwrap_or(self.base_position + self.buffer.len());
        let consumed = retained_from.saturating_sub(self.base_position);
        self.buffer.drain(..consumed);
        if let Some(memory) = &self.memory {
            memory.shrink(consumed);
        }
        self.base_position = retained_from;
    }
}

impl Iterator for Reader<'_> {
    type Item = ExtendedPointId;

    fn next(&mut self) -> Option<Self::Item> {
        let mut coordinator = self.coordinator.borrow_mut();
        let position = coordinator.reader_positions[self.reader]
            .expect("an active rank-stream reader has a position");
        let buffered_end = coordinator.base_position + coordinator.buffer.len();
        if position == buffered_end && !coordinator.exhausted {
            match coordinator
                .inner
                .as_mut()
                .expect("an unfinished shared stream has a producer")
                .next()
            {
                Some(identity) => {
                    coordinator.buffer.push_back(identity);
                    if let Some(memory) = &coordinator.memory {
                        memory.grow(1);
                    }
                    coordinator.physical_pulls += 1;
                    coordinator.peak_buffered_identities = coordinator
                        .peak_buffered_identities
                        .max(coordinator.buffer.len());
                }
                None => {
                    coordinator.exhausted = true;
                    coordinator.inner = None;
                }
            }
        }
        let identity = position
            .checked_sub(coordinator.base_position)
            .and_then(|offset| coordinator.buffer.get(offset))
            .copied();
        if identity.is_some() {
            coordinator.reader_positions[self.reader] = Some(position + 1);
            coordinator.prune_consumed();
        }
        identity
    }
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        let mut coordinator = self.coordinator.borrow_mut();
        coordinator.reader_positions[self.reader] = None;
        coordinator.prune_consumed();
        if coordinator.reader_positions.iter().all(Option::is_none) {
            coordinator.cancelled |= !coordinator.exhausted;
            coordinator.inner = None;
            coordinator.exhausted = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream<I>(values: I) -> ExactRrfStream<'static>
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: 'static,
    {
        Box::new(values.into_iter().map(ExtendedPointId::from))
    }

    #[test]
    fn readers_share_physical_pulls_and_keep_independent_positions() {
        let shared = SharedRankStream::new(stream([1, 2, 3]));
        let mut first = shared.subscribe();
        let mut second = shared.subscribe();

        assert_eq!(first.next(), Some(1.into()));
        assert_eq!(first.next(), Some(2.into()));
        assert_eq!(second.next(), Some(1.into()));
        assert_eq!(second.next(), Some(2.into()));
        assert_eq!(second.next(), Some(3.into()));
        assert_eq!(first.next(), Some(3.into()));

        let telemetry = shared.telemetry();
        assert_eq!(telemetry.physical_pulls, 3);
        assert_eq!(telemetry.buffered_identities, 0);
        assert_eq!(telemetry.peak_buffered_identities, 2);
    }

    #[test]
    fn dropping_the_last_reader_cancels_the_producer() {
        let shared = SharedRankStream::new(stream(0..100));
        let mut reader = shared.subscribe();
        assert_eq!(reader.next(), Some(0.into()));

        drop(reader);

        let telemetry = shared.telemetry();
        assert!(telemetry.cancelled);
        assert!(telemetry.exhausted);
        assert_eq!(telemetry.active_readers, 0);
    }
}
