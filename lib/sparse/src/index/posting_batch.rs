//! Shared document-batch scoring kernel for native Sparse executors.

use common::types::{PointOffsetType, ScoreType};

use super::posting_list_common::PostingListIter;
use crate::common::types::DimWeight;

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
