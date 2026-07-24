//! Shared document-batch scoring kernel for native Sparse executors.

use common::types::{PointOffsetType, ScoreType};

use super::posting_list_common::PostingListIter;
use crate::common::types::DimWeight;

#[cfg(feature = "stratumind-research")]
#[derive(Default)]
pub(crate) struct TouchedScoreBuffer {
    scores: Vec<ScoreType>,
    epochs: Vec<u32>,
    touched: Vec<usize>,
    epoch: u32,
}

#[cfg(feature = "stratumind-research")]
impl TouchedScoreBuffer {
    pub(crate) fn clear(&mut self) {
        self.touched.clear();
    }

    pub(crate) fn begin(&mut self, len: usize) {
        if self.scores.len() < len {
            self.scores.resize(len, 0.0);
            self.epochs.resize(len, 0);
        }
        self.touched.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epochs.fill(0);
            self.epoch = 1;
        }
    }

    #[inline(always)]
    fn add(&mut self, local_id: usize, score: ScoreType) {
        if self.epochs[local_id] != self.epoch {
            self.epochs[local_id] = self.epoch;
            self.scores[local_id] = score;
            self.touched.push(local_id);
        } else {
            self.scores[local_id] += score;
        }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (usize, ScoreType)> + '_ {
        self.touched
            .iter()
            .map(|&local_id| (local_id, self.scores[local_id]))
    }

    pub(crate) fn touched_len(&self) -> usize {
        self.touched.len()
    }
}

/// Accumulates one inclusive document-id batch from weighted posting iterators.
///
/// Callers own scheduling, filtering, and result queues. Keeping accumulation
/// here makes ordinary Qdrant Top-K search and Stratumind's certified cursor
/// share the same numerical hot path.
pub(crate) fn score_posting_batch<'a, T, P>(
    postings: P,
    batch_start_id: PointOffsetType,
    batch_last_id: PointOffsetType,
    scores: &mut Vec<ScoreType>,
) -> usize
where
    T: PostingListIter + 'a,
    P: IntoIterator<Item = (&'a mut T, DimWeight)>,
{
    let batch_len = batch_last_id - batch_start_id + 1;
    scores.clear();
    scores.resize(batch_len as usize, 0.0);

    let mut posting_elements_visited = 0;
    for (posting, query_weight) in postings {
        posting.skip_to(batch_start_id);
        let elements_before = posting.len_to_end();
        posting.for_each_till_id(
            batch_last_id,
            scores.as_mut_slice(),
            #[inline(always)]
            |scores, id, weight| {
                let local_id = (id - batch_start_id) as usize;
                // SAFETY: `for_each_till_id` only yields ids in this batch.
                *unsafe { scores.get_unchecked_mut(local_id) } += weight * query_weight;
            },
        );
        posting_elements_visited += elements_before - posting.len_to_end();
    }
    posting_elements_visited
}

/// Scores an inclusive document-id range without mutating the posting cursors.
///
/// Compressed postings override range traversal with direct physical chunk
/// lookup, while mutable postings use a binary search over their element slice.
#[cfg(feature = "stratumind-research")]
pub(crate) fn score_posting_range_dense<'a, T, P>(
    postings: P,
    batch_start_id: PointOffsetType,
    batch_last_id: PointOffsetType,
    scores: &mut Vec<ScoreType>,
) -> usize
where
    T: PostingListIter + 'a,
    P: IntoIterator<Item = (&'a T, DimWeight)>,
{
    let batch_len = batch_last_id - batch_start_id + 1;
    scores.clear();
    scores.resize(batch_len as usize, 0.0);

    let mut posting_elements_visited = 0;
    for (posting, query_weight) in postings {
        posting_elements_visited += posting.for_each_in_id_range(
            batch_start_id,
            batch_last_id,
            scores.as_mut_slice(),
            #[inline(always)]
            |scores, id, weight| {
                let local_id = (id - batch_start_id) as usize;
                *unsafe { scores.get_unchecked_mut(local_id) } += weight * query_weight;
            },
        );
    }
    posting_elements_visited
}

#[cfg(feature = "stratumind-research")]
pub(crate) fn score_posting_range_touched<'a, T, P>(
    postings: P,
    batch_start_id: PointOffsetType,
    batch_last_id: PointOffsetType,
    scores: &mut TouchedScoreBuffer,
) -> usize
where
    T: PostingListIter + 'a,
    P: IntoIterator<Item = (&'a T, DimWeight)>,
{
    let batch_len = batch_last_id - batch_start_id + 1;
    scores.begin(batch_len as usize);

    let mut posting_elements_visited = 0;
    for (posting, query_weight) in postings {
        posting_elements_visited += posting.for_each_in_id_range(
            batch_start_id,
            batch_last_id,
            scores,
            #[inline(always)]
            |scores, id, weight| {
                let local_id = (id - batch_start_id) as usize;
                scores.add(local_id, weight * query_weight);
            },
        );
    }
    posting_elements_visited
}
