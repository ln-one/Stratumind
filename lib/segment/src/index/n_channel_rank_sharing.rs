// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Legacy N-channel query deduplication and physical stream sharing.
//!
//! The V4 composition core uses `rank_stream`; this module remains isolated so
//! the earlier query-specific experiments do not become a dependency of the
//! generic DAG executor.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use sparse::common::sparse_vector::RemappedSparseVector;

use super::n_channel_exact::ExactChannelQuery;
use crate::common::reciprocal_rank_fusion::ExactRrfStream;
use crate::types::ExtendedPointId;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RankStreamSharingTelemetry {
    pub logical_streams: usize,
    pub physical_streams: usize,
    pub shared_groups: usize,
    pub logical_pulls: usize,
    pub physical_pulls: usize,
    pub buffered_identities: usize,
}

struct Coordinator<'a> {
    inner: Option<ExactRrfStream<'a>>,
    buffer: VecDeque<ExtendedPointId>,
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

impl<'a> SharedRankStream<'a> {
    pub fn new(inner: ExactRrfStream<'a>) -> Self {
        Self {
            coordinator: Rc::new(RefCell::new(Coordinator {
                inner: Some(inner),
                buffer: VecDeque::new(),
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

pub(super) struct RankStreamSharingObserver<'a> {
    group_members: Vec<Vec<usize>>,
    shared: Vec<(usize, Rc<RefCell<Coordinator<'a>>>)>,
}

impl RankStreamSharingObserver<'_> {
    pub fn telemetry(&self, source_pulls: &[usize]) -> RankStreamSharingTelemetry {
        let mut physical_pulls = 0usize;
        let mut buffered_identities = 0usize;
        for (group, members) in self.group_members.iter().enumerate() {
            if members.len() == 1 {
                physical_pulls += source_pulls[members[0]];
            } else {
                let coordinator = self
                    .shared
                    .iter()
                    .find_map(|(candidate, coordinator)| {
                        (*candidate == group).then_some(coordinator)
                    })
                    .expect("every shared rank group has a coordinator")
                    .borrow();
                physical_pulls += coordinator.physical_pulls;
                buffered_identities += coordinator.peak_buffered_identities;
            }
        }
        RankStreamSharingTelemetry {
            logical_streams: source_pulls.len(),
            physical_streams: self.group_members.len(),
            shared_groups: self.shared.len(),
            logical_pulls: source_pulls.iter().sum(),
            physical_pulls,
            buffered_identities,
        }
    }
}

pub(super) fn share_identical_rank_streams<'a>(
    queries: &[ExactChannelQuery<'_>],
    sources: Vec<ExactRrfStream<'a>>,
    weights: Option<&[f32]>,
    share_dense: bool,
    share_sparse: bool,
) -> (Vec<ExactRrfStream<'a>>, RankStreamSharingObserver<'a>) {
    assert_eq!(queries.len(), sources.len());
    let group_ids = exact_duplicate_group_ids(queries, weights, share_dense, share_sparse);
    let group_count = group_ids.iter().copied().max().map_or(0, |group| group + 1);
    let mut group_members = vec![Vec::<usize>::new(); group_count];
    for (source, &group) in group_ids.iter().enumerate() {
        group_members[group].push(source);
    }

    let mut sources = sources.into_iter().map(Some).collect::<Vec<_>>();
    let mut logical_sources = std::iter::repeat_with(|| None)
        .take(sources.len())
        .collect::<Vec<Option<ExactRrfStream<'a>>>>();
    let mut shared = Vec::new();
    for (group, members) in group_members.iter().enumerate() {
        if members.len() == 1 {
            let source = members[0];
            logical_sources[source] = sources[source].take();
            continue;
        }
        let representative = members[0];
        let shared_stream = SharedRankStream::new(
            sources[representative]
                .take()
                .expect("rank stream representative is present"),
        );
        for &source in members {
            sources[source].take();
            logical_sources[source] = Some(shared_stream.subscribe());
        }
        shared.push((group, Rc::clone(&shared_stream.coordinator)));
    }

    (
        logical_sources
            .into_iter()
            .map(|source| source.expect("every logical rank stream is restored"))
            .collect(),
        RankStreamSharingObserver {
            group_members,
            shared,
        },
    )
}

pub(super) fn unique_sparse_queries(
    queries: &[ExactChannelQuery<'_>],
    weights: Option<&[f32]>,
) -> Vec<RemappedSparseVector> {
    let group_ids = exact_duplicate_group_ids(queries, weights, false, true);
    queries
        .iter()
        .enumerate()
        .filter_map(|(source, query)| {
            let is_representative = !group_ids[..source].contains(&group_ids[source]);
            match (is_representative, query) {
                (true, ExactChannelQuery::Sparse(query)) => Some(query.clone()),
                _ => None,
            }
        })
        .collect()
}

pub(super) fn dense_representatives(
    queries: &[ExactChannelQuery<'_>],
    weights: Option<&[f32]>,
) -> Vec<bool> {
    let group_ids = exact_duplicate_group_ids(queries, weights, true, false);
    group_ids
        .iter()
        .enumerate()
        .map(|(source, group)| !group_ids[..source].contains(group))
        .collect()
}

fn queries_are_bitwise_identical(
    left: &ExactChannelQuery<'_>,
    right: &ExactChannelQuery<'_>,
) -> bool {
    match (left, right) {
        (ExactChannelQuery::Dense(left), ExactChannelQuery::Dense(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(*right)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        (ExactChannelQuery::Sparse(left), ExactChannelQuery::Sparse(right)) => {
            left.indices == right.indices
                && left.values.len() == right.values.len()
                && left
                    .values
                    .iter()
                    .zip(&right.values)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        _ => false,
    }
}

fn exact_duplicate_group_ids(
    queries: &[ExactChannelQuery<'_>],
    weights: Option<&[f32]>,
    share_dense: bool,
    share_sparse: bool,
) -> Vec<usize> {
    let mut representatives = Vec::<usize>::new();
    queries
        .iter()
        .enumerate()
        .map(|(source, query)| {
            if let Some(group) = representatives.iter().position(|&representative| {
                sharing_enabled(query, share_dense, share_sparse)
                    && queries_are_bitwise_identical(query, &queries[representative])
                    && source_weight(weights, source).to_bits()
                        == source_weight(weights, representative).to_bits()
            }) {
                group
            } else {
                representatives.push(source);
                representatives.len() - 1
            }
        })
        .collect()
}

fn sharing_enabled(query: &ExactChannelQuery<'_>, dense: bool, sparse: bool) -> bool {
    match query {
        ExactChannelQuery::Dense(_) => dense,
        ExactChannelQuery::Sparse(_) => sparse,
    }
}

fn source_weight(weights: Option<&[f32]>, source: usize) -> f32 {
    weights
        .and_then(|weights| weights.get(source))
        .copied()
        .unwrap_or(1.0)
}

#[cfg(test)]
mod tests {
    use sparse::common::sparse_vector::RemappedSparseVector;

    use super::*;

    #[test]
    fn identical_sources_share_physical_pulls_without_changing_logical_order() {
        let query = RemappedSparseVector {
            indices: vec![1],
            values: vec![2.0],
        };
        let queries = vec![
            ExactChannelQuery::Sparse(query.clone()),
            ExactChannelQuery::Sparse(query),
        ];
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new([1_u64, 2, 3].into_iter().map(ExtendedPointId::from)),
            Box::new([1_u64, 2, 3].into_iter().map(ExtendedPointId::from)),
        ];
        let (mut sources, observer) =
            share_identical_rank_streams(&queries, sources, None, false, true);
        let first: Vec<_> = sources[0].by_ref().collect();
        let second: Vec<_> = sources[1].by_ref().collect();
        assert_eq!(first, second);
        let telemetry = observer.telemetry(&[3, 3]);
        assert_eq!(telemetry.logical_pulls, 6);
        assert_eq!(telemetry.physical_pulls, 3);
        assert_eq!(telemetry.physical_streams, 1);
        assert_eq!(telemetry.buffered_identities, 3);
    }

    #[test]
    fn identical_queries_with_different_weights_remain_independent() {
        let query = RemappedSparseVector {
            indices: vec![1],
            values: vec![2.0],
        };
        let queries = vec![
            ExactChannelQuery::Sparse(query.clone()),
            ExactChannelQuery::Sparse(query),
        ];
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new([1_u64].into_iter().map(ExtendedPointId::from)),
            Box::new([1_u64].into_iter().map(ExtendedPointId::from)),
        ];
        let weights = [1.0, 2.0];
        let (mut sources, observer) =
            share_identical_rank_streams(&queries, sources, Some(&weights), false, true);
        assert!(sources[0].next().is_some());
        assert!(sources[1].next().is_some());
        let telemetry = observer.telemetry(&[1, 1]);
        assert_eq!(telemetry.physical_streams, 2);
        assert_eq!(telemetry.physical_pulls, 2);
        assert_eq!(telemetry.shared_groups, 0);
    }

    #[test]
    fn identical_dense_queries_share_when_enabled() {
        let query = [1.0f32, -0.5];
        let queries = vec![
            ExactChannelQuery::Dense(&query),
            ExactChannelQuery::Dense(&query),
        ];
        assert_eq!(dense_representatives(&queries, None), vec![true, false]);
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new([1_u64, 3].into_iter().map(ExtendedPointId::from)),
            Box::new(std::iter::empty()),
        ];
        let (mut sources, observer) =
            share_identical_rank_streams(&queries, sources, None, true, false);
        assert_eq!(
            sources[0].by_ref().collect::<Vec<_>>(),
            vec![1_u64.into(), 3_u64.into()]
        );
        assert_eq!(
            sources[1].by_ref().collect::<Vec<_>>(),
            vec![1_u64.into(), 3_u64.into()]
        );
        let telemetry = observer.telemetry(&[2, 2]);
        assert_eq!(telemetry.physical_streams, 1);
        assert_eq!(telemetry.physical_pulls, 2);
    }

    #[test]
    fn replay_buffer_reclaims_the_prefix_consumed_by_every_reader() {
        let query = RemappedSparseVector {
            indices: vec![1],
            values: vec![2.0],
        };
        let queries = vec![
            ExactChannelQuery::Sparse(query.clone()),
            ExactChannelQuery::Sparse(query),
        ];
        let sources: Vec<ExactRrfStream<'_>> = vec![
            Box::new([1_u64, 2, 3].into_iter().map(ExtendedPointId::from)),
            Box::new(std::iter::empty()),
        ];
        let (mut sources, observer) =
            share_identical_rank_streams(&queries, sources, None, false, true);

        for expected in [1_u64, 2, 3] {
            assert_eq!(sources[0].next(), Some(expected.into()));
            assert_eq!(sources[1].next(), Some(expected.into()));
        }

        let coordinator = observer.shared[0].1.borrow();
        assert!(coordinator.buffer.is_empty());
        assert_eq!(coordinator.base_position, 3);
        assert_eq!(coordinator.peak_buffered_identities, 1);
    }
}
