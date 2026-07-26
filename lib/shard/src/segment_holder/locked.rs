use std::sync::Arc;
use std::time::Duration;

use parking_lot::lock_api::ArcRwLockReadGuard;
use parking_lot::{
    RawRwLock, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard,
};

use crate::segment_holder::SegmentHolder;

/// Exclusive guard for a Segment-generation mutation.
///
/// Updates, snapshot transitions, and optimizer publish/rollback use this side
/// of the barrier.
#[must_use = "dropping this guard immediately releases the Segment generation barrier"]
#[allow(dead_code)] // Field is held for its RAII Drop behavior, not for reading
pub struct SegmentUpdateGuard<'a>(RwLockWriteGuard<'a, ()>);

/// Owned shared guard that pins one Segment generation for a resumable read.
///
/// The guard is owned so it can travel with an ExactRankSession without
/// borrowing `LockedSegmentHolder`. Several exact sessions can coexist.
#[must_use = "dropping this guard releases the pinned Segment generation"]
#[allow(dead_code)] // Field is held for its RAII Drop behavior, not for reading
pub struct SegmentGenerationGuard(ArcRwLockReadGuard<RawRwLock, ()>);

#[derive(Clone, Debug)]
pub struct LockedSegmentHolder {
    holder: Arc<RwLock<SegmentHolder>>,
    /// Shared-read/exclusive-write barrier around Segment generations.
    ///
    /// It stays external to `holder`: readers pin a generation without
    /// retaining the holder lock, while update and maintenance transitions
    /// retain their existing exclusive serialization.
    generation_barrier: Arc<RwLock<()>>,
}

impl LockedSegmentHolder {
    pub fn new(segment_holder: SegmentHolder) -> Self {
        Self {
            holder: Arc::new(RwLock::new(segment_holder)),
            generation_barrier: Arc::new(RwLock::new(())),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, SegmentHolder> {
        self.holder.read()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, SegmentHolder> {
        self.holder.write()
    }

    pub fn upgradable_read(&self) -> RwLockUpgradableReadGuard<'_, SegmentHolder> {
        self.holder.upgradable_read()
    }

    pub fn try_read_for(&self, timeout: Duration) -> Option<RwLockReadGuard<'_, SegmentHolder>> {
        self.holder.try_read_for(timeout)
    }

    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, SegmentHolder>> {
        self.holder.try_read()
    }

    /// Acquire exclusive authority to mutate or replace the Segment generation.
    ///
    /// Acquire before a holder read/write lock.
    pub fn acquire_update_guard(&self) -> SegmentUpdateGuard<'_> {
        SegmentUpdateGuard(self.generation_barrier.write())
    }

    /// Pin the current Segment generation for a long-lived resumable reader.
    ///
    /// Acquire before cloning handles under the holder read lock.
    pub fn acquire_generation_guard(&self) -> SegmentGenerationGuard {
        SegmentGenerationGuard(self.generation_barrier.read_arc())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn generation_readers_share_and_delay_update_transition() {
        let holder = LockedSegmentHolder::new(SegmentHolder::default());
        let first_reader = holder.acquire_generation_guard();
        let second_reader = holder.acquire_generation_guard();

        let writer_holder = holder.clone();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _guard = writer_holder.acquire_update_guard();
            acquired_tx.send(()).unwrap();
        });

        assert!(
            acquired_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "writer must wait while either generation reader is alive"
        );

        drop(first_reader);
        assert!(
            acquired_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "readers must not be serialized into one exclusive guard"
        );

        drop(second_reader);
        acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("writer must proceed after the last generation reader exits");
        writer.join().unwrap();
    }
}
