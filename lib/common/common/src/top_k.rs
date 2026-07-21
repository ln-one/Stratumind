use std::cmp::{Ordering, Reverse};

use ordered_float::{Float, OrderedFloat};

use crate::types::{ScoreType, ScoredPointOffset};

/// TopK implementation following the median algorithm described in
/// <https://quickwit.io/blog/top-k-complexity>.
///
/// Keeps the largest `k` ScoredPointOffset.
#[derive(Default)]
pub struct TopK {
    k: usize,
    elements: Vec<Reverse<RankedPoint>>,
    threshold: Option<RankedPoint>,
}

/// Total order used by sparse top-k selection.
///
/// A larger score is better. Equal scores are resolved by the smaller point
/// offset, which gives callers a deterministic order instead of inheriting
/// `select_nth_unstable`'s arbitrary tie handling.
#[derive(Copy, Clone, Debug, PartialEq)]
struct RankedPoint(ScoredPointOffset);

impl Eq for RankedPoint {}

impl Ord for RankedPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.0.score)
            .cmp(&OrderedFloat(other.0.score))
            .then_with(|| other.0.idx.cmp(&self.0.idx))
    }
}

impl PartialOrd for RankedPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            elements: Vec::with_capacity(2 * k),
            threshold: None,
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns the minimum score of the top k elements.
    ///
    /// Updated every 2k elements.
    /// Initially set to `ScoreType::MIN`.
    pub fn threshold(&self) -> ScoreType {
        self.threshold
            .map(|point| point.0.score)
            .unwrap_or_else(ScoreType::min_value)
    }

    /// Whether a point can still enter the retained top-k under the complete
    /// score-and-identity order.
    pub fn would_accept(&self, element: ScoredPointOffset) -> bool {
        self.k != 0
            && !element.score.is_nan()
            && self
                .threshold
                .is_none_or(|threshold| RankedPoint(element) > threshold)
    }

    pub fn push(&mut self, element: ScoredPointOffset) {
        if self.would_accept(element) {
            self.elements.push(Reverse(RankedPoint(element)));
            // check if full
            if self.elements.len() == self.k * 2 {
                let (_, median_el, _) = self.elements.select_nth_unstable(self.k - 1);
                self.threshold = Some(median_el.0);
                self.elements.truncate(self.k);
            }
        }
    }

    pub fn into_vec(mut self) -> Vec<ScoredPointOffset> {
        self.elements.sort_unstable();
        self.elements.truncate(self.k);
        self.elements
            .into_iter()
            .map(|Reverse(point)| point.0)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn empty_with_double_capacity() {
        let top_k = TopK::new(3);
        assert_eq!(top_k.len(), 0);
        assert_eq!(top_k.elements.capacity(), 2 * 3);
        assert_eq!(top_k.threshold(), ScoreType::MIN);
    }

    #[test]
    fn test_top_k_under() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].score, 2.0);
        assert_eq!(res[1].score, 1.0);
    }

    #[test]
    fn test_top_k_over() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.len(), 1);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 3.0, idx: 3 });
        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        top_k.push(ScoredPointOffset { score: 4.0, idx: 4 });
        assert_eq!(top_k.len(), 4);
        assert_eq!(top_k.threshold(), ScoreType::MIN);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].score, 4.0);
        assert_eq!(res[1].score, 3.0);
        assert_eq!(res[2].score, 2.0);
    }

    #[test]
    fn test_top_k_pruned() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 4.0, idx: 4 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 3);

        top_k.push(ScoredPointOffset { score: 5.0, idx: 5 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 4);

        top_k.push(ScoredPointOffset { score: 3.0, idx: 3 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 5);

        top_k.push(ScoredPointOffset { score: 6.0, idx: 6 });
        assert_eq!(top_k.threshold(), 4.0);
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.elements.capacity(), 6);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].score, 6.0);
        assert_eq!(res[1].score, 5.0);
        assert_eq!(res[2].score, 4.0);
    }

    #[test]
    fn test_top_same_scores() {
        let mut top_k = TopK::new(3);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 1);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 4 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 2);

        top_k.push(ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 3);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 5 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 4);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 3 });
        assert_eq!(top_k.threshold(), ScoreType::MIN);
        assert_eq!(top_k.len(), 5);

        top_k.push(ScoredPointOffset { score: 1.0, idx: 6 });
        assert_eq!(top_k.threshold(), 1.0);
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k.elements.capacity(), 6);

        let res = top_k.into_vec();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], ScoredPointOffset { score: 2.0, idx: 2 });
        assert_eq!(res[1], ScoredPointOffset { score: 1.0, idx: 1 });
        assert_eq!(res[2], ScoredPointOffset { score: 1.0, idx: 3 });
    }

    #[test]
    fn later_smaller_identity_replaces_equal_score_threshold() {
        let mut top_k = TopK::new(2);
        for idx in [9, 8, 7, 6, 5, 4, 3, 2, 1] {
            top_k.push(ScoredPointOffset { score: 1.0, idx });
        }

        assert_eq!(
            top_k.into_vec(),
            vec![
                ScoredPointOffset { score: 1.0, idx: 1 },
                ScoredPointOffset { score: 1.0, idx: 2 },
            ]
        );
    }

    #[test]
    fn zero_k_discards_everything() {
        let mut top_k = TopK::new(0);
        top_k.push(ScoredPointOffset { score: 1.0, idx: 1 });

        assert!(top_k.into_vec().is_empty());
    }

    #[test]
    fn nan_score_is_ignored() {
        let mut top_k = TopK::new(2);
        top_k.push(ScoredPointOffset {
            score: f32::NAN,
            idx: 1,
        });
        top_k.push(ScoredPointOffset { score: 1.0, idx: 2 });

        assert_eq!(
            top_k.into_vec(),
            vec![ScoredPointOffset { score: 1.0, idx: 2 }]
        );
    }
}
