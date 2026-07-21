use std::cmp::{Ordering, max, min};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

use common::counter::hardware_counter::HardwareCounterCell;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use common::universal_io::Result;
use serde::Serialize;

use super::posting_batch::score_posting_batch;
use super::posting_list_common::PostingListIter;
use crate::SearchScratch;
use crate::common::sparse_vector::{RemappedSparseVector, score_vectors};
use crate::common::types::{DimId, DimWeight};
use crate::index::inverted_index::InvertedIndex;
use crate::index::posting_list::PostingListIterator;

// Stratumind extension: executor-level access telemetry for same-kernel experiments.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct SearchTelemetry {
    pub posting_lists: usize,
    pub posting_elements: usize,
    pub posting_elements_visited: usize,
    pub posting_elements_remaining: usize,
    pub posting_elements_skipped: usize,
    pub batch_count: usize,
    pub scored_id_span: usize,
    pub prune_attempts: usize,
    pub prune_successes: usize,
    pub block_prune_attempts: usize,
    pub block_prune_successes: usize,
    pub block_pruning_router_disabled: bool,
    pub block_pruning_trend_disabled: bool,
    pub use_pruning: bool,
    pub use_block_pruning: bool,
    pub cancelled: bool,
}

/// Iterator over posting lists with a reference to the corresponding query index and weight
pub struct IndexedPostingListIterator<T: PostingListIter> {
    posting_list_iterator: T,
    query_index: DimId,
    query_weight: DimWeight,
}

/// Making this larger makes the search faster but uses more (pooled) memory
const ADVANCE_BATCH_SIZE: usize = 10_000;

pub struct SearchContext<'a, T: PostingListIter = PostingListIterator<'a>> {
    postings_iterators: Vec<IndexedPostingListIterator<T>>,
    query: RemappedSparseVector,
    top: usize,
    is_stopped: &'a AtomicBool,
    top_results: TopK,
    min_record_id: Option<PointOffsetType>, // min_record_id ids across all posting lists
    max_record_id: PointOffsetType,         // max_record_id ids across all posting lists
    /// Scores buffer from [`SearchScratch`].
    scores: &'a mut Vec<ScoreType>,
    use_pruning: bool,
    use_block_pruning: bool,
    block_prune_failure_limit: Option<usize>,
    consecutive_block_prune_failures: usize,
    block_prune_has_succeeded: bool,
    hardware_counter: &'a HardwareCounterCell,
    telemetry: SearchTelemetry,
    batch_size: PointOffsetType,
}

impl<'a, T: PostingListIter> SearchContext<'a, T> {
    pub fn new(
        query: RemappedSparseVector,
        top: usize,
        inverted_index: &'a impl InvertedIndex<Iter<'a> = T>,
        scratch: &'a mut SearchScratch<'_>,
        is_stopped: &'a AtomicBool,
        hardware_counter: &'a HardwareCounterCell,
    ) -> Result<SearchContext<'a, T>> {
        let mut postings_iterators = Vec::new();
        // track min and max record ids across all posting lists
        let mut max_record_id = 0;
        let mut min_record_id = u32::MAX;
        // iterate over query indices
        for (query_weight_offset, id) in query.indices.iter().enumerate() {
            let mut it = inverted_index.get(*id, &scratch.arena, hardware_counter)?;
            if let (Some(first), Some(last_id)) = (it.peek(), it.last_id()) {
                // check if new min
                let min_record_id_posting = first.record_id;
                min_record_id = min(min_record_id, min_record_id_posting);

                // check if new max
                let max_record_id_posting = last_id;
                max_record_id = max(max_record_id, max_record_id_posting);

                // capture query info
                let query_index = *id;
                let query_weight = query.values[query_weight_offset];

                postings_iterators.push(IndexedPostingListIterator {
                    posting_list_iterator: it,
                    query_index,
                    query_weight,
                });
            }
        }
        let top_results = TopK::new(top);
        // Query vectors with negative values can NOT use the pruning mechanism which relies on the pre-computed `max_next_weight`.
        // The max contribution per posting list that we calculate is not made to compute the max value of two negative numbers.
        // This is a limitation of the current pruning implementation.
        let use_pruning = T::reliable_max_next_weight() && query.values.iter().all(|v| *v >= 0.0);
        let use_block_pruning =
            T::reliable_block_max() && query.values.iter().all(|value| *value >= 0.0);
        let min_record_id = Some(min_record_id);
        let telemetry = SearchTelemetry {
            posting_lists: postings_iterators.len(),
            posting_elements: postings_iterators
                .iter()
                .map(|posting| posting.posting_list_iterator.len_to_end())
                .sum(),
            use_pruning,
            use_block_pruning,
            ..Default::default()
        };
        Ok(SearchContext {
            postings_iterators,
            query,
            top,
            is_stopped,
            top_results,
            min_record_id,
            max_record_id,
            scores: &mut scratch.scores,
            use_pruning,
            use_block_pruning,
            block_prune_failure_limit: None,
            consecutive_block_prune_failures: 0,
            block_prune_has_succeeded: false,
            hardware_counter,
            telemetry,
            batch_size: ADVANCE_BATCH_SIZE as PointOffsetType,
        })
    }

    const DEFAULT_SCORE: f32 = 0.0;

    /// Plain search against the given ids without any pruning
    pub fn plain_search(&mut self, ids: &[PointOffsetType]) -> Vec<ScoredPointOffset> {
        // sort ids to fully leverage posting list iterator traversal
        let mut sorted_ids = ids.to_vec();
        sorted_ids.sort_unstable();

        let cpu_counter = self.hardware_counter.cpu_counter();

        let mut indices = Vec::with_capacity(self.query.indices.len());
        let mut values = Vec::with_capacity(self.query.values.len());
        for id in sorted_ids {
            // check for cancellation
            if self.is_stopped.load(Relaxed) {
                self.telemetry.cancelled = true;
                break;
            }

            self.telemetry.scored_id_span += 1;

            indices.clear();
            values.clear();
            // collect indices and values for the current record id from the query's posting lists *only*
            for posting_iterator in self.postings_iterators.iter_mut() {
                // rely on underlying binary search as the posting lists are sorted by record id
                match posting_iterator.posting_list_iterator.skip_to(id) {
                    None => {} // no match for posting list
                    Some(element) => {
                        self.telemetry.posting_elements_visited += 1;
                        // match for posting list
                        indices.push(posting_iterator.query_index);
                        values.push(element.weight);
                    }
                }
            }

            if values.is_empty() {
                continue;
            }

            // Accumulate the sum of the length of the retrieved sparse vector and the query vector length
            // as measurement for CPU usage of plain search.
            cpu_counter
                .incr_delta(self.query.indices.len() + values.len() * size_of::<DimWeight>());

            // reconstruct sparse vector and score against query
            let sparse_score =
                score_vectors(&indices, &values, &self.query.indices, &self.query.values)
                    .unwrap_or(Self::DEFAULT_SCORE);

            self.top_results.push(ScoredPointOffset {
                score: sparse_score,
                idx: id,
            });
        }
        let top = std::mem::take(&mut self.top_results);
        top.into_vec()
    }

    /// Advance posting lists iterators in a batch fashion.
    fn advance_batch<F: Fn(PointOffsetType) -> bool>(
        &mut self,
        batch_start_id: PointOffsetType,
        batch_last_id: PointOffsetType,
        filter_condition: &F,
    ) {
        // init batch scores
        let batch_len = batch_last_id - batch_start_id + 1;
        self.telemetry.batch_count += 1;
        self.telemetry.scored_id_span += batch_len as usize;
        self.telemetry.posting_elements_visited += score_posting_batch(
            self.postings_iterators
                .iter_mut()
                .map(|posting| (&mut posting.posting_list_iterator, posting.query_weight)),
            batch_start_id,
            batch_last_id,
            self.scores,
        );

        for (local_index, &score) in self.scores.iter().enumerate() {
            if score != 0.0 {
                let score_point_offset = ScoredPointOffset {
                    score,
                    idx: batch_start_id + local_index as PointOffsetType,
                };
                // Publish only points that can beat the current complete
                // score-and-identity threshold.
                if !self.top_results.would_accept(score_point_offset) {
                    continue;
                }
                // do not score if filter condition is not satisfied
                if !filter_condition(score_point_offset.idx) {
                    continue;
                }
                self.top_results.push(score_point_offset);
            }
        }
    }

    /// Compute scores for the last posting list quickly
    fn process_last_posting_list<F: Fn(PointOffsetType) -> bool>(&mut self, filter_condition: &F) {
        debug_assert_eq!(self.postings_iterators.len(), 1);
        let posting = &mut self.postings_iterators[0];
        let elements_before = posting.posting_list_iterator.len_to_end();
        posting.posting_list_iterator.for_each_till_id(
            PointOffsetType::MAX,
            &mut (),
            |_, id, weight| {
                // do not score if filter condition is not satisfied
                if !filter_condition(id) {
                    return;
                }
                let score = weight * posting.query_weight;
                self.top_results.push(ScoredPointOffset { score, idx: id });
            },
        );
        self.telemetry.posting_elements_visited +=
            elements_before - posting.posting_list_iterator.len_to_end();
    }

    /// Returns the next min record id from all posting list iterators
    ///
    /// returns None if all posting list iterators are exhausted
    fn next_min_id(to_inspect: &mut [IndexedPostingListIterator<T>]) -> Option<PointOffsetType> {
        let mut min_record_id = None;

        // Iterate to find min record id at the head of the posting lists
        for posting_iterator in to_inspect.iter_mut() {
            if let Some(next_element) = posting_iterator.posting_list_iterator.peek() {
                match min_record_id {
                    None => min_record_id = Some(next_element.record_id), // first record with matching id
                    Some(min_id_seen) => {
                        // update min record id if smaller
                        if next_element.record_id < min_id_seen {
                            min_record_id = Some(next_element.record_id);
                        }
                    }
                }
            }
        }

        min_record_id
    }

    /// Make sure the longest posting list is at the head of the posting list iterators
    pub(crate) fn promote_longest_posting_lists_to_the_front(&mut self) {
        // find index of longest posting list
        let posting_index = self
            .postings_iterators
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.posting_list_iterator
                    .len_to_end()
                    .cmp(&b.posting_list_iterator.len_to_end())
            })
            .map(|(index, _)| index);

        if let Some(posting_index) = posting_index {
            // make sure it is not already at the head
            if posting_index != 0 {
                // swap longest posting list to the head
                self.postings_iterators.swap(0, posting_index);
            }
        }
    }

    /// How many elements are left in the posting list iterator
    #[cfg(test)]
    pub(crate) fn posting_list_len(&self, idx: usize) -> usize {
        self.postings_iterators[idx]
            .posting_list_iterator
            .len_to_end()
    }

    /// Search for the top k results that satisfy the filter condition
    pub fn search<F: Fn(PointOffsetType) -> bool>(
        &mut self,
        filter_condition: &F,
    ) -> Vec<ScoredPointOffset> {
        if self.postings_iterators.is_empty() {
            return Vec::new();
        }

        {
            // Measure CPU usage of indexed sparse search.
            // Assume the complexity of the search as total volume of the posting lists
            // that are traversed in the batched search.
            let mut cpu_cost = 0;

            for posting in self.postings_iterators.iter() {
                cpu_cost += posting.posting_list_iterator.len_to_end()
                    * posting.posting_list_iterator.element_size();
            }
            self.hardware_counter.cpu_counter().incr_delta(cpu_cost);
        }

        let mut best_min_score = f32::MIN;
        loop {
            // check for cancellation (atomic amortized by batch)
            if self.is_stopped.load(Relaxed) {
                self.telemetry.cancelled = true;
                break;
            }

            // prepare next iterator of batched ids
            let Some(start_batch_id) = self.min_record_id else {
                break;
            };

            // compute batch range of contiguous ids for the next batch
            let last_batch_id = min(
                start_batch_id.saturating_add(self.batch_size),
                self.max_record_id,
            );

            if self.prune_batch_if_safe(start_batch_id, last_batch_id) {
                self.postings_iterators.retain(|posting_iterator| {
                    posting_iterator.posting_list_iterator.len_to_end() != 0
                });
                self.min_record_id = Self::next_min_id(&mut self.postings_iterators);
                continue;
            }

            // advance and score posting lists iterators
            self.advance_batch(start_batch_id, last_batch_id, filter_condition);

            // remove empty posting lists if necessary
            self.postings_iterators.retain(|posting_iterator| {
                posting_iterator.posting_list_iterator.len_to_end() != 0
            });

            // update min_record_id
            self.min_record_id = Self::next_min_id(&mut self.postings_iterators);

            // check if all posting lists are exhausted
            if self.postings_iterators.is_empty() {
                break;
            }

            // if only one posting list left, we can score it quickly
            if self.postings_iterators.len() == 1 && !self.use_block_pruning {
                self.process_last_posting_list(filter_condition);
                break;
            }

            // we potentially have enough results to prune low performing posting lists
            if self.use_pruning && self.top_results.len() >= self.top {
                // current min score
                let new_min_score = self.top_results.threshold();
                if new_min_score == best_min_score {
                    // no improvement in lowest best score since last pruning - skip pruning
                    continue;
                } else {
                    best_min_score = new_min_score;
                }
                // make sure the first posting list is the longest for pruning
                self.promote_longest_posting_lists_to_the_front();

                // prune posting list that cannot possibly contribute to the top results
                let pruned = self.prune_longest_posting_list(new_min_score);
                if pruned {
                    // update min_record_id
                    self.min_record_id = Self::next_min_id(&mut self.postings_iterators);
                }
            }
        }
        // posting iterators exhausted, return result queue
        let queue = std::mem::take(&mut self.top_results);
        queue.into_vec()
    }

    fn prune_batch_if_safe(
        &mut self,
        start_batch_id: PointOffsetType,
        last_batch_id: PointOffsetType,
    ) -> bool {
        if !self.use_block_pruning || self.top_results.len() < self.top {
            return false;
        }
        self.telemetry.block_prune_attempts += 1;
        let mut upper_bound = 0.0;
        for posting in &mut self.postings_iterators {
            if let Some(max_weight) = posting
                .posting_list_iterator
                .max_weight_till_id(last_batch_id)
            {
                upper_bound += max_weight * posting.query_weight;
            }
        }
        if self.top_results.would_accept(ScoredPointOffset {
            score: upper_bound,
            idx: start_batch_id,
        }) {
            if !self.block_prune_has_succeeded {
                self.consecutive_block_prune_failures += 1;
                if self
                    .block_prune_failure_limit
                    .is_some_and(|limit| self.consecutive_block_prune_failures >= limit)
                {
                    self.use_block_pruning = false;
                    self.telemetry.use_block_pruning = false;
                    self.telemetry.block_pruning_router_disabled = true;
                }
            }
            return false;
        }

        self.block_prune_has_succeeded = true;
        self.consecutive_block_prune_failures = 0;
        for posting in &mut self.postings_iterators {
            let position_before = posting.posting_list_iterator.current_index();
            posting.posting_list_iterator.skip_till_id(last_batch_id);
            self.telemetry.posting_elements_skipped +=
                posting.posting_list_iterator.current_index() - position_before;
        }
        self.telemetry.block_prune_successes += 1;
        true
    }

    /// Prune posting lists that cannot possibly contribute to the top results
    /// Assumes longest posting list is at the head of the posting list iterators
    /// Returns true if the longest posting list was pruned
    pub fn prune_longest_posting_list(&mut self, min_score: f32) -> bool {
        if self.postings_iterators.is_empty() {
            return false;
        }
        self.telemetry.prune_attempts += 1;
        // peek first element of longest posting list
        let (longest_posting_iterator, rest_iterators) = self.postings_iterators.split_at_mut(1);
        let longest_posting_iterator = &mut longest_posting_iterator[0];
        if let Some(element) = longest_posting_iterator.posting_list_iterator.peek() {
            let next_min_id_in_others = Self::next_min_id(rest_iterators);
            match next_min_id_in_others {
                Some(next_min_id) => {
                    match next_min_id.cmp(&element.record_id) {
                        Ordering::Equal => {
                            // if the next min id in the other posting lists is the same as the current one,
                            // we can't prune the current element as it needs to be scored properly across posting lists
                            return false;
                        }
                        Ordering::Less => {
                            // we can't prune as there the other posting lists contains smaller smaller ids that need to scored first
                            return false;
                        }
                        Ordering::Greater => {
                            // next_min_id is > element.record_id there is a chance to prune up to `next_min_id`
                            // check against the max possible score using the `max_next_weight`
                            // we can under prune as we should actually check the best score up to `next_min_id` - 1 only
                            // instead of the max possible score but it is not possible to know the best score up to `next_min_id` - 1
                            let max_weight_from_list = element.weight.max(element.max_next_weight);
                            let max_score_contribution =
                                max_weight_from_list * longest_posting_iterator.query_weight;
                            // Equality is not enough to prune under the stable
                            // score-and-identity order: the skipped range may
                            // contain a smaller identity with the same score.
                            if max_score_contribution < min_score {
                                // prune to next_min_id
                                let longest_posting_iterator =
                                    &mut self.postings_iterators[0].posting_list_iterator;
                                let position_before_pruning =
                                    longest_posting_iterator.current_index();
                                longest_posting_iterator.skip_to(next_min_id);
                                let position_after_pruning =
                                    longest_posting_iterator.current_index();
                                // check if pruning took place
                                let skipped = position_after_pruning - position_before_pruning;
                                if skipped != 0 {
                                    self.telemetry.prune_successes += 1;
                                    self.telemetry.posting_elements_skipped += skipped;
                                    return true;
                                }
                                return false;
                            }
                        }
                    }
                }
                None => {
                    // the current posting list is the only one left, we can potentially skip it to the end
                    // check against the max possible score using the `max_next_weight`
                    let max_weight_from_list = element.weight.max(element.max_next_weight);
                    let max_score_contribution =
                        max_weight_from_list * longest_posting_iterator.query_weight;
                    // Keep equal-score suffixes because their identities may
                    // still beat the current K-th identity.
                    if max_score_contribution < min_score {
                        // prune to the end!
                        let longest_posting_iterator = &mut self.postings_iterators[0];
                        let position_before_pruning = longest_posting_iterator
                            .posting_list_iterator
                            .current_index();
                        longest_posting_iterator.posting_list_iterator.skip_to_end();
                        let position_after_pruning = longest_posting_iterator
                            .posting_list_iterator
                            .current_index();
                        self.telemetry.prune_successes += 1;
                        self.telemetry.posting_elements_skipped +=
                            position_after_pruning - position_before_pruning;
                        return true;
                    }
                }
            }
        }
        // no pruning took place
        false
    }

    pub fn telemetry(&self) -> SearchTelemetry {
        let posting_elements_remaining = self
            .postings_iterators
            .iter()
            .map(|posting| posting.posting_list_iterator.len_to_end())
            .sum();
        let mut telemetry = self.telemetry;
        telemetry.posting_elements_remaining = posting_elements_remaining;
        telemetry
    }

    /// Enable or disable compressed-posting block pruning for controlled
    /// same-index experiments. Enabling cannot override an unsupported iterator
    /// or a query with negative weights.
    pub fn set_block_pruning(&mut self, enabled: bool) {
        self.use_block_pruning &= enabled;
        self.telemetry.use_block_pruning = self.use_block_pruning;
    }

    pub fn block_pruning_enabled(&self) -> bool {
        self.use_block_pruning
    }

    /// Disable max-next-weight posting pruning for strict-execution ablations.
    /// Enabling cannot override an iterator or query that does not support it.
    pub fn set_posting_pruning(&mut self, enabled: bool) {
        self.use_pruning &= enabled;
        self.telemetry.use_pruning = self.use_pruning;
    }

    /// Stop paying for block-bound checks after `limit` consecutive failures.
    /// This changes only the physical executor; disabling pruning preserves the
    /// same exhaustive sparse Top-K contract.
    pub fn set_block_prune_failure_limit(&mut self, limit: Option<usize>) {
        assert!(
            limit != Some(0),
            "block prune failure limit must be positive"
        );
        self.block_prune_failure_limit = limit;
    }

    /// Disable block pruning when the query-weighted last posting blocks have a
    /// larger envelope than the first blocks. This is a cost hint only; falling
    /// back to ordinary posting traversal preserves exact Top-K.
    pub fn apply_block_max_endpoint_trend_router(&mut self) {
        if !self.use_block_pruning {
            return;
        }
        let mut first_upper_bound = 0.0;
        let mut last_upper_bound = 0.0;
        for posting in &self.postings_iterators {
            let Some((first, last)) = posting.posting_list_iterator.block_max_endpoints() else {
                return;
            };
            first_upper_bound += first * posting.query_weight;
            last_upper_bound += last * posting.query_weight;
        }
        if last_upper_bound >= first_upper_bound {
            self.use_block_pruning = false;
            self.telemetry.use_block_pruning = false;
            self.telemetry.block_pruning_trend_disabled = true;
        }
    }

    /// Set the contiguous document-id batch width used by the sparse scorer.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        assert!(batch_size > 0, "sparse batch size must be positive");
        self.batch_size = batch_size
            .try_into()
            .expect("sparse batch size exceeds PointOffsetType");
    }
}
