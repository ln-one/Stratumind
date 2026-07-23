use std::borrow::Cow;

use super::*;
use crate::SearchScratchArena;
use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use crate::index::rank_probe::RankProbeTarget;

#[test]
fn native_kernel_stream_matches_exhaustive_order_without_rescan() {
    let mut builder = InvertedIndexBuilder::new();
    let mut documents = Vec::new();
    for id in 0..30_000_u32 {
        let vector = RemappedSparseVector {
            indices: vec![0, 1, 2],
            values: vec![
                1.0 + (id % 11) as f32,
                100.0 - id as f32 / 400.0,
                (id % 17) as f32 / 3.0,
            ],
        };
        builder.add(id, vector.clone());
        documents.push((id, vector));
    }
    let temp = tempfile::tempdir().unwrap();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        temp.path(),
    )
    .unwrap();
    let query = RemappedSparseVector {
        indices: vec![0, 1, 2],
        values: vec![1.0, 2.0, 0.5],
    };
    let mut expected: Vec<_> = documents
        .iter()
        .filter_map(|(id, vector)| {
            vector
                .score(&query)
                .filter(|score| *score != 0.0)
                .map(|score| ScoredPointOffset { idx: *id, score })
        })
        .collect();
    expected.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.idx.cmp(&right.idx))
    });

    let arena = SearchScratchArena::new_slow();
    let hardware_counter = HardwareCounterCell::disposable();
    let stopped = AtomicBool::new(false);
    let mut stream =
        NativeSearchContextRankStream::new(query, 257, &index, &arena, &hardware_counter).unwrap();

    let first: Vec<_> = (0..20)
        .map(|_| stream.next_result(&stopped).unwrap().unwrap())
        .collect();
    assert_eq!(first, expected[..20]);
    let visited_after_pause = stream.telemetry().posting_elements_visited;
    assert!(visited_after_pause > 0);

    let mut actual = first;
    while let Some(point) = stream.next_result(&stopped).unwrap() {
        actual.push(point);
    }
    assert_eq!(actual, expected);
    assert_eq!(
        stream.telemetry().posting_elements_visited,
        stream.telemetry().query_posting_elements
    );
    assert!(stream.telemetry().posting_elements_visited >= visited_after_pause);
}

#[test]
fn native_kernel_stream_closes_flat_ties_at_eof_and_cancellation_is_sticky() {
    let mut builder = InvertedIndexBuilder::new();
    for id in 0..257_u32 {
        builder.add(
            id,
            RemappedSparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
        );
    }
    let temp = tempfile::tempdir().unwrap();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        temp.path(),
    )
    .unwrap();
    let arena = SearchScratchArena::new_slow();
    let hardware_counter = HardwareCounterCell::disposable();
    let stopped = AtomicBool::new(false);
    let query = RemappedSparseVector {
        indices: vec![0],
        values: vec![1.0],
    };
    let mut stream =
        NativeSearchContextRankStream::new(query.clone(), 31, &index, &arena, &hardware_counter)
            .unwrap();
    let actual: Vec<_> = std::iter::from_fn(|| stream.next_result(&stopped).unwrap()).collect();
    assert_eq!(
        actual.iter().map(|point| point.idx).collect::<Vec<_>>(),
        (0..257).collect::<Vec<_>>()
    );

    let mut cancelled =
        NativeSearchContextRankStream::new(query, 31, &index, &arena, &hardware_counter).unwrap();
    let cancelled_flag = AtomicBool::new(true);
    assert_eq!(
        cancelled.next_result(&cancelled_flag),
        Err(NativeSparseCursorError::Cancelled)
    );
    cancelled_flag.store(false, Relaxed);
    assert_eq!(
        cancelled.next_result(&cancelled_flag),
        Err(NativeSparseCursorError::Cancelled)
    );
}

#[test]
fn native_kernel_publishes_monotone_batch_certificates_before_next_rank_is_fixed() {
    let mut builder = InvertedIndexBuilder::new();
    let mut documents = Vec::new();
    for id in 0..1_000_u32 {
        let vector = RemappedSparseVector {
            indices: vec![0, 1],
            values: vec![(id % 31) as f32 + 1.0, (1_000 - id) as f32 / 10.0],
        };
        builder.add(id, vector.clone());
        documents.push((id, vector));
    }
    let temp = tempfile::tempdir().unwrap();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        temp.path(),
    )
    .unwrap();
    let arena = SearchScratchArena::new_slow();
    let hardware_counter = HardwareCounterCell::disposable();
    let stopped = AtomicBool::new(false);
    let query = RemappedSparseVector {
        indices: vec![0, 1],
        values: vec![1.0, 1.0],
    };
    let mut expected: Vec<_> = documents
        .iter()
        .map(|(id, vector)| ScoredPointOffset {
            idx: *id,
            score: vector.score(&query).unwrap(),
        })
        .collect();
    expected.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.idx.cmp(&right.idx))
    });
    let true_ranks: std::collections::HashMap<_, _> = expected
        .iter()
        .enumerate()
        .map(|(rank, point)| (point.idx, rank))
        .collect();
    let mut stream =
        NativeSearchContextRankStream::new(query, 37, &index, &arena, &hardware_counter).unwrap();

    assert_eq!(
        stream.rank_certificate(),
        SparseIncrementalRankCertificate {
            exact_prefix: Vec::new(),
            candidates: Vec::new(),
            unnamed_rank_floor: 0,
            exhausted: false,
        }
    );
    let mut previous_prefix = Vec::new();
    while stream.advance_certificate(&stopped).unwrap() {
        let certificate = stream.rank_certificate();
        assert!(certificate.exact_prefix.starts_with(&previous_prefix));
        if certificate.exhausted {
            assert_eq!(certificate.unnamed_rank_floor, usize::MAX);
        } else {
            assert_eq!(
                certificate.unnamed_rank_floor,
                certificate.exact_prefix.len()
            );
        }
        assert!(certificate.candidates.iter().all(|candidate| {
            candidate.min_rank >= certificate.exact_prefix.len()
                && candidate.max_rank >= candidate.min_rank
        }));
        assert_eq!(
            certificate.exact_prefix,
            expected
                .iter()
                .take(certificate.exact_prefix.len())
                .map(|point| point.idx)
                .collect::<Vec<_>>()
        );
        let named: std::collections::HashSet<_> = certificate
            .exact_prefix
            .iter()
            .copied()
            .chain(
                certificate
                    .candidates
                    .iter()
                    .map(|candidate| candidate.point),
            )
            .collect();
        for candidate in &certificate.candidates {
            let rank = true_ranks[&candidate.point];
            assert!((candidate.min_rank..=candidate.max_rank).contains(&rank));
        }
        if !certificate.exhausted {
            assert!(true_ranks.iter().all(|(point, rank)| {
                named.contains(point) || *rank >= certificate.unnamed_rank_floor
            }));
        }
        previous_prefix = certificate.exact_prefix;
    }
    let certificate = stream.rank_certificate();
    assert!(certificate.exhausted);
    assert!(certificate.candidates.is_empty());
    assert_eq!(certificate.exact_prefix.len(), 1_000);
}

#[test]
fn qdrant_kernel_batch_probes_exact_ranks_for_known_competitors() {
    let mut builder = InvertedIndexBuilder::new();
    let mut documents = Vec::new();
    for id in 0..20_000_u32 {
        let vector = RemappedSparseVector {
            indices: vec![0, 1, 2],
            values: vec![
                200.0 - id as f32 / 100.0,
                (id % 97) as f32 / 7.0,
                (id % 13) as f32,
            ],
        };
        builder.add(id, vector.clone());
        documents.push((id, vector));
    }
    let temp = tempfile::tempdir().unwrap();
    let index = InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
        Cow::Owned(builder.build()),
        temp.path(),
    )
    .unwrap();
    let query = RemappedSparseVector {
        indices: vec![0, 1, 2],
        values: vec![1.0, 0.5, 0.25],
    };
    let mut expected: Vec<_> = documents
        .iter()
        .map(|(id, vector)| ScoredPointOffset {
            idx: *id,
            score: vector.score(&query).unwrap(),
        })
        .collect();
    expected.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.idx.cmp(&right.idx))
    });
    let target_ranks = [0, 7, 31, 255, 2_047];
    let targets: Vec<_> = target_ranks
        .iter()
        .map(|&rank| RankProbeTarget {
            idx: expected[rank].idx,
            score: expected[rank].score,
        })
        .collect();
    let stopped = AtomicBool::new(false);
    let hardware_counter = HardwareCounterCell::disposable();
    let mut scratch = SearchScratch::new_for_test();
    let mut context = SearchContext::new(
        query,
        targets.len(),
        &index,
        &mut scratch,
        &stopped,
        &hardware_counter,
    )
    .unwrap();
    context.set_batch_size(257);

    let actual = context.probe_exact_ranks(&targets, &|_| true).unwrap();

    assert_eq!(
        actual.iter().map(|probe| probe.rank).collect::<Vec<_>>(),
        target_ranks
    );
    assert!(context.telemetry().block_prune_successes > 0);
}
