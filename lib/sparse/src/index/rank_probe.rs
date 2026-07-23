use std::collections::HashSet;
use std::sync::atomic::Ordering::Relaxed;

use common::types::{PointOffsetType, ScoreType};

use super::posting_block_stream::NativeSparseCursorError;
use super::posting_list_common::PostingListIter;
use super::search_context::SearchContext;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankProbeTarget {
    pub idx: PointOffsetType,
    pub score: ScoreType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRankProbe {
    pub idx: PointOffsetType,
    pub rank: usize,
}

impl<T: PostingListIter> SearchContext<'_, T> {
    /// Resolve exact channel ranks for already-known competitors in one DAAT
    /// pass. This adapter reuses Qdrant posting iterators and block metadata;
    /// it does not implement a second Sparse scorer.
    pub fn probe_exact_ranks<F: Fn(PointOffsetType) -> bool>(
        &mut self,
        targets: &[RankProbeTarget],
        filter_condition: &F,
    ) -> std::result::Result<Vec<ExactRankProbe>, NativeSparseCursorError> {
        assert!(
            targets.iter().all(|target| target.score.is_finite())
                && targets
                    .iter()
                    .map(|target| target.idx)
                    .collect::<HashSet<_>>()
                    .len()
                    == targets.len(),
            "Sparse exact-rank probes require unique finite targets"
        );
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let mut ranks = vec![0_usize; targets.len()];
        if self.postings_iterators.is_empty() {
            return Ok(targets
                .iter()
                .map(|target| ExactRankProbe {
                    idx: target.idx,
                    rank: 0,
                })
                .collect());
        }

        loop {
            if self.is_stopped.load(Relaxed) {
                self.telemetry.cancelled = true;
                return Err(NativeSparseCursorError::Cancelled);
            }
            let Some(batch_start_id) = self.min_record_id else {
                break;
            };
            let batch_last_id = batch_start_id
                .saturating_add(self.batch_size.saturating_sub(1))
                .min(self.max_record_id);
            if self.prune_rank_probe_batch_if_safe(batch_start_id, batch_last_id, targets) {
                self.postings_iterators
                    .retain(|posting| posting.posting_list_iterator.len_to_end() != 0);
                self.min_record_id = Self::next_min_id(&mut self.postings_iterators);
                continue;
            }

            let batch_len = batch_last_id - batch_start_id + 1;
            self.telemetry.batch_count += 1;
            self.telemetry.scored_id_span += batch_len as usize;
            self.scores.clear();
            self.scores.resize(batch_len as usize, 0.0);
            for posting in &mut self.postings_iterators {
                let elements_before = posting.posting_list_iterator.len_to_end();
                posting.posting_list_iterator.for_each_till_id(
                    batch_last_id,
                    self.scores.as_mut_slice(),
                    #[inline(always)]
                    |scores, id, weight| {
                        let local_id = (id - batch_start_id) as usize;
                        *unsafe { scores.get_unchecked_mut(local_id) } +=
                            weight * posting.query_weight;
                    },
                );
                self.telemetry.posting_elements_visited +=
                    elements_before - posting.posting_list_iterator.len_to_end();
            }
            for (offset, &score) in self.scores.iter().enumerate() {
                if score == 0.0 {
                    continue;
                }
                let idx = batch_start_id + offset as PointOffsetType;
                if !filter_condition(idx) {
                    continue;
                }
                for (target, rank) in targets.iter().zip(&mut ranks) {
                    if score > target.score || (score == target.score && idx < target.idx) {
                        *rank += 1;
                    }
                }
            }
            self.postings_iterators
                .retain(|posting| posting.posting_list_iterator.len_to_end() != 0);
            self.min_record_id = Self::next_min_id(&mut self.postings_iterators);
        }
        Ok(targets
            .iter()
            .zip(ranks)
            .map(|(target, rank)| ExactRankProbe {
                idx: target.idx,
                rank,
            })
            .collect())
    }

    fn prune_rank_probe_batch_if_safe(
        &mut self,
        batch_start_id: PointOffsetType,
        batch_last_id: PointOffsetType,
        targets: &[RankProbeTarget],
    ) -> bool {
        if !self.use_block_pruning {
            return false;
        }
        self.telemetry.block_prune_attempts += 1;
        let mut upper_bound = 0.0_f32;
        for posting in &mut self.postings_iterators {
            let Some(max_weight) = posting
                .posting_list_iterator
                .max_weight_till_id(batch_last_id)
            else {
                continue;
            };
            let contribution = max_weight * posting.query_weight;
            if contribution.is_nan() || contribution == f32::INFINITY {
                return false;
            }
            upper_bound += contribution.max(0.0);
        }
        if !upper_bound.is_finite() {
            return false;
        }
        let upper_bound = next_up_nonnegative(upper_bound);
        let cannot_outrank_any_target = targets.iter().all(|target| {
            upper_bound < target.score
                || (upper_bound == target.score && batch_start_id > target.idx)
        });
        if !cannot_outrank_any_target {
            return false;
        }
        for posting in &mut self.postings_iterators {
            let position_before = posting.posting_list_iterator.current_index();
            posting.posting_list_iterator.skip_till_id(batch_last_id);
            self.telemetry.posting_elements_skipped +=
                posting.posting_list_iterator.current_index() - position_before;
        }
        self.telemetry.block_prune_successes += 1;
        true
    }
}

fn next_up_nonnegative(value: f32) -> f32 {
    debug_assert!(value >= 0.0 && value.is_finite());
    f32::from_bits(value.to_bits() + 1)
}
