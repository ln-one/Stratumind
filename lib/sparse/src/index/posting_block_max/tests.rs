use std::borrow::Cow;
use std::sync::Arc;

use super::*;
use crate::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;

#[test]
fn state_resumes_across_workers_and_matches_full_order() {
    let mut builder = InvertedIndexBuilder::new();
    for id in 0..257u32 {
        builder.add(
            id,
            RemappedSparseVector {
                indices: vec![0],
                values: vec![257.0 - id as f32],
            },
        );
    }
    let temp = tempfile::tempdir().unwrap();
    let index = Arc::new(
        InvertedIndexCompressedImmutableRam::<f32>::from_ram_index(
            Cow::Owned(builder.build()),
            temp.path(),
        )
        .unwrap(),
    );
    let arena = SearchScratchArena::new_slow();
    let hardware_counter = HardwareCounterCell::new();
    let state = PostingBlockMaxState::new(
        index.as_ref(),
        RemappedSparseVector {
            indices: vec![0],
            values: vec![1.0],
        },
        32,
        &arena,
        &hardware_counter,
    )
    .unwrap();

    let first_index = index.clone();
    let first = std::thread::spawn(move || {
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::new();
        let mut state = state;
        let batch = state
            .next_batch_with(
                first_index.as_ref(),
                &arena,
                &hardware_counter,
                7,
                &AtomicBool::new(false),
            )
            .unwrap();
        (state, batch)
    })
    .join()
    .unwrap();

    let second = std::thread::spawn(move || {
        let arena = SearchScratchArena::new_slow();
        let hardware_counter = HardwareCounterCell::new();
        let (mut state, mut points) = first;
        loop {
            let batch = state
                .next_batch_with(
                    index.as_ref(),
                    &arena,
                    &hardware_counter,
                    31,
                    &AtomicBool::new(false),
                )
                .unwrap();
            if batch.is_empty() {
                break;
            }
            points.extend(batch);
        }
        points
    })
    .join()
    .unwrap();

    assert_eq!(second.len(), 257);
    assert_eq!(
        second.iter().map(|point| point.idx).collect::<Vec<_>>(),
        (0..257).collect::<Vec<_>>(),
    );
}
