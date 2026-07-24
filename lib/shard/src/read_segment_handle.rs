// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Short-lived read access over an owned Segment handle.
//!
//! Selectively adapted from Qdrant edge `ReadSegmentHandle` at
//! `842f701aa` (original introduction `f70a1462f`). ExactRankSession only
//! needs the mutable-shard [`LockedSegment`] implementation; the read-only
//! follower and its file-opening lifecycle deliberately remain outside this
//! production execution path.

use parking_lot::RwLockReadGuard;
use segment::entry::ReadSegmentEntry;

use crate::locked_segment::LockedSegment;

/// An owned handle that acquires a Segment read guard for one bounded task.
pub(crate) trait ReadSegmentHandle: Send + Sync {
    type Segment: ReadSegmentEntry + ?Sized;

    fn read_segment(&self) -> RwLockReadGuard<'_, Self::Segment>;
}

impl ReadSegmentHandle for LockedSegment {
    type Segment = dyn ReadSegmentEntry;

    fn read_segment(&self) -> RwLockReadGuard<'_, Self::Segment> {
        self.get_read().read()
    }
}
