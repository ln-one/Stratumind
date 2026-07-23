//! Adaptive wrapper that routes search `spawn_blocking` calls between two
//! pre-built Tokio runtimes — one with a smaller blocking pool for CPU-bound
//! load and one sized via [`common::defaults::search_thread_count`] for
//! IO-bound load. The choice
//! is re-evaluated lazily on the spawn path, at most once per
//! [`ADJUST_INTERVAL`].
//!
//! - Start in [`SearchMode::HighIo`] (over-committed) so cold starts make
//!   progress on IO without waiting for a CPU sample.
//! - Switch to [`SearchMode::HighCpu`] when the process is over
//!   [`HIGH_CPU_THRESHOLD`] of total cores — extra threads just thrash.
//! - Switch back to [`SearchMode::HighIo`] when CPU drops below
//!   [`LOW_CPU_THRESHOLD`] — search is likely IO-bound and more parallelism
//!   helps.
//!
//! The hot path is two relaxed atomic loads (mode + last-adjust timestamp)
//! plus the same `Handle::spawn_blocking` call as before — no semaphore.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::process_cpu_usage::{CPU_USAGE_WINDOW, process_cpu_usage_cores};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

/// Minimum interval between mode re-evaluations. Matches the CPU sampling
/// window so each adjustment sees a fresh reading.
const ADJUST_INTERVAL: Duration = CPU_USAGE_WINDOW;

/// Switch to [`SearchMode::HighCpu`] when the process uses more than this
/// fraction of all CPUs.
const HIGH_CPU_THRESHOLD: f32 = 0.9;

/// Switch back to [`SearchMode::HighIo`] when the process uses less than
/// this fraction of all CPUs.
const LOW_CPU_THRESHOLD: f32 = 0.5;

/// Active search runtime — encoded in [`Inner::mode`] as a `u8`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// CPU-bound: route to the runtime with the smaller blocking pool.
    HighCpu,
    /// IO-bound: route to the runtime sized via [`common::defaults::search_thread_count`].
    HighIo,
}

impl SearchMode {
    fn as_u8(self) -> u8 {
        match self {
            SearchMode::HighCpu => 0,
            SearchMode::HighIo => 1,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => SearchMode::HighCpu,
            _ => SearchMode::HighIo,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::HighCpu => "high_cpu",
            SearchMode::HighIo => "high_io",
        }
    }
}

#[derive(Clone)]
pub struct AdaptiveSearchHandle {
    inner: Arc<Inner>,
}

struct Inner {
    high_cpu: Handle,
    high_io: Handle,
    /// Number of logical CPUs, read once at construction. Used as the
    /// denominator when converting process CPU usage to a utilization ratio.
    num_cpus: usize,
    /// Current mode (encoded via [`SearchMode::as_u8`]).
    mode: AtomicU8,
    /// Monotonic ns (relative to [`Inner::start`]) at which the mode was
    /// last re-evaluated. Initialized so the first adjust is suppressed
    /// until [`ADJUST_INTERVAL`] elapses — [`process_cpu_usage_cores`]
    /// returns `None` until it has two samples anyway.
    last_adjust_ns: AtomicU64,
    /// Reference instant for `last_adjust_ns`.
    start: Instant,
    high_cpu_session_slots: Arc<Semaphore>,
    high_io_session_slots: Arc<Semaphore>,
    high_cpu_session_capacity: usize,
    high_io_session_capacity: usize,
}

struct ReservedSessionCapacity {
    _permit: OwnedSemaphorePermit,
    slots: usize,
    spawned: AtomicUsize,
}

/// One atomically capacity-reserved blocking session on a fixed Qdrant search
/// runtime. Clones share the reservation and may spawn at most `slots` tasks
/// in total. The caller computes the complete query task set before reserving.
#[derive(Clone)]
pub(crate) struct ReservedSearchSession {
    handle: Handle,
    _capacity: Arc<ReservedSessionCapacity>,
}

/// All capacity needed by one exact query, with cursor workers and their
/// coordinator allowed to use different pre-built Qdrant search runtimes.
///
/// A combined reservation is preferred. If the active pool fits every cursor
/// but not the extra coordinator, the coordinator may use one atomically
/// reserved slot from the other *distinct* Qdrant runtime. No task is spawned
/// until both reservations exist.
pub(crate) struct ReservedExactSearchSession {
    workers: ReservedSearchSession,
    coordinator: ReservedSearchSession,
    worker_slots: usize,
}

impl ReservedExactSearchSession {
    pub(crate) fn workers(&self) -> ReservedSearchSession {
        self.workers.clone()
    }

    pub(crate) fn coordinator(&self) -> ReservedSearchSession {
        self.coordinator.clone()
    }

    pub(crate) fn worker_slots(&self) -> usize {
        self.worker_slots
    }
}

impl ReservedSearchSession {
    pub(crate) fn spawn_blocking<F, R>(&self, task: F) -> Option<JoinHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let mut spawned = self._capacity.spawned.load(Ordering::Relaxed);
        loop {
            if spawned >= self._capacity.slots {
                return None;
            }
            match self._capacity.spawned.compare_exchange_weak(
                spawned,
                spawned + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let capacity = self._capacity.clone();
                    return Some(self.handle.spawn_blocking(move || {
                        // Keep the physical reservation alive until the
                        // blocking task itself exits, even if its async caller
                        // is cancelled and drops the JoinHandle.
                        let _capacity = capacity;
                        task()
                    }));
                }
                Err(current) => spawned = current,
            }
        }
    }
}

// =============================================================================
// Production API
// =============================================================================

impl AdaptiveSearchHandle {
    /// Build from the two pre-constructed search runtimes.
    pub fn new(high_cpu: Handle, high_io: Handle) -> Self {
        let high_cpu_capacity = common::cpu::get_num_cpus().max(1);
        let high_io_capacity = common::defaults::search_thread_count(high_cpu_capacity).max(1);
        Self::new_with_session_capacities(high_cpu, high_io, high_cpu_capacity, high_io_capacity)
    }

    /// Build with the exact blocking capacities configured on both runtimes.
    /// Exact long-lived sessions use these values for all-or-nothing query
    /// reservations; ordinary short `spawn_blocking` calls remain unchanged.
    pub fn new_with_session_capacities(
        high_cpu: Handle,
        high_io: Handle,
        high_cpu_capacity: usize,
        high_io_capacity: usize,
    ) -> Self {
        let num_cpus = common::cpu::get_num_cpus().max(1);
        let high_cpu_capacity = high_cpu_capacity.max(1);
        let high_io_capacity = high_io_capacity.max(1);
        let inner = Arc::new(Inner {
            high_cpu,
            high_io,
            num_cpus,
            mode: AtomicU8::new(SearchMode::HighIo.as_u8()),
            last_adjust_ns: AtomicU64::new(0),
            start: Instant::now(),
            high_cpu_session_slots: Arc::new(Semaphore::new(high_cpu_capacity)),
            high_io_session_slots: Arc::new(Semaphore::new(high_io_capacity)),
            high_cpu_session_capacity: high_cpu_capacity,
            high_io_session_capacity: high_io_capacity,
        });
        Self { inner }
    }

    /// Spawn a blocking closure on the search runtime selected by the
    /// current mode. Before spawning, attempt a lazy mode re-evaluation
    /// rate-limited to once per [`ADJUST_INTERVAL`].
    pub fn spawn_blocking<F, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.maybe_adjust();
        self.handle_for_current_mode().spawn_blocking(f)
    }

    /// Currently active mode.
    pub fn current_mode(&self) -> SearchMode {
        SearchMode::from_u8(self.inner.mode.load(Ordering::Relaxed))
    }

    /// Reserve every long-lived cursor worker plus its coordinator before
    /// starting an exact query.
    ///
    /// The common path keeps all tasks on the active runtime. When the active
    /// pool can hold all cursor workers but the additional coordinator would
    /// be the only overflow, the coordinator is placed on the other Qdrant
    /// search runtime. This avoids making a normal multi-Segment collection
    /// miss the native plan solely because `workers + 1` crosses the active
    /// pool boundary, while preserving all-or-nothing startup.
    pub(crate) fn try_reserve_exact_workers(
        &self,
        worker_slots: usize,
    ) -> Option<ReservedExactSearchSession> {
        if worker_slots == 0 || worker_slots >= u32::MAX as usize {
            return None;
        }
        self.maybe_adjust();
        let primary = self.current_mode();
        let total_slots = worker_slots.checked_add(1)?;
        if let Some(combined) = self.try_reserve_exact_on_mode(primary, total_slots) {
            log::debug!(
                "exact search session reserved {worker_slots} workers plus coordinator on {} runtime",
                primary.as_str(),
            );
            return Some(ReservedExactSearchSession {
                workers: combined.clone(),
                coordinator: combined,
                worker_slots,
            });
        }

        let secondary = match primary {
            SearchMode::HighCpu => SearchMode::HighIo,
            SearchMode::HighIo => SearchMode::HighCpu,
        };
        if self.handle_for_mode(primary).id() == self.handle_for_mode(secondary).id() {
            log::debug!(
                "exact search session reservation miss: {worker_slots} workers plus coordinator exceed the single runtime capacity",
            );
            return None;
        }
        if let Some(combined) = self.try_reserve_exact_on_mode(secondary, total_slots) {
            log::debug!(
                "exact search session reserved {worker_slots} workers plus coordinator on fallback {} runtime",
                secondary.as_str(),
            );
            return Some(ReservedExactSearchSession {
                workers: combined.clone(),
                coordinator: combined,
                worker_slots,
            });
        }

        for (worker_mode, coordinator_mode) in [(primary, secondary), (secondary, primary)] {
            let Some(workers) = self.try_reserve_exact_on_mode(worker_mode, worker_slots) else {
                continue;
            };
            let Some(coordinator) = self.try_reserve_exact_on_mode(coordinator_mode, 1) else {
                continue;
            };
            log::debug!(
                "exact search session reserved {worker_slots} workers on {} runtime and coordinator on {} runtime",
                worker_mode.as_str(),
                coordinator_mode.as_str(),
            );
            return Some(ReservedExactSearchSession {
                workers,
                coordinator,
                worker_slots,
            });
        }

        log::debug!(
            "exact search session reservation miss: {worker_slots} workers plus coordinator do not fit; {} capacity/available={}/{}, {} capacity/available={}/{}",
            primary.as_str(),
            self.capacity_for_mode(primary),
            self.semaphore_for_mode(primary).available_permits(),
            secondary.as_str(),
            self.capacity_for_mode(secondary),
            self.semaphore_for_mode(secondary).available_permits(),
        );
        None
    }

    fn try_reserve_exact_on_mode(
        &self,
        mode: SearchMode,
        slots: usize,
    ) -> Option<ReservedSearchSession> {
        let (handle, semaphore, capacity) = match mode {
            SearchMode::HighCpu => (
                self.inner.high_cpu.clone(),
                Arc::clone(&self.inner.high_cpu_session_slots),
                self.inner.high_cpu_session_capacity,
            ),
            SearchMode::HighIo => (
                self.inner.high_io.clone(),
                Arc::clone(&self.inner.high_io_session_slots),
                self.inner.high_io_session_capacity,
            ),
        };
        if slots > capacity {
            return None;
        }
        let permit = semaphore.try_acquire_many_owned(slots as u32).ok()?;
        Some(ReservedSearchSession {
            handle,
            _capacity: Arc::new(ReservedSessionCapacity {
                _permit: permit,
                slots,
                spawned: AtomicUsize::new(0),
            }),
        })
    }

    /// Tokio handle for the currently active mode. Exposed for the rare
    /// caller that needs to `.enter()` the runtime context (e.g. snapshot
    /// creation that internally calls `tokio::task::spawn_blocking`).
    pub fn tokio_handle(&self) -> &Handle {
        self.handle_for_current_mode()
    }

    fn handle_for_current_mode(&self) -> &Handle {
        self.handle_for_mode(self.current_mode())
    }

    fn handle_for_mode(&self, mode: SearchMode) -> &Handle {
        match mode {
            SearchMode::HighCpu => &self.inner.high_cpu,
            SearchMode::HighIo => &self.inner.high_io,
        }
    }

    fn semaphore_for_mode(&self, mode: SearchMode) -> &Semaphore {
        match mode {
            SearchMode::HighCpu => &self.inner.high_cpu_session_slots,
            SearchMode::HighIo => &self.inner.high_io_session_slots,
        }
    }

    fn capacity_for_mode(&self, mode: SearchMode) -> usize {
        match mode {
            SearchMode::HighCpu => self.inner.high_cpu_session_capacity,
            SearchMode::HighIo => self.inner.high_io_session_capacity,
        }
    }

    /// Attempt to re-evaluate the active mode. No-op if we adjusted less
    /// than [`ADJUST_INTERVAL`] ago or if a CPU sample is unavailable.
    fn maybe_adjust(&self) {
        let now_ns = self.inner.start.elapsed().as_nanos() as u64;
        let interval_ns = ADJUST_INTERVAL.as_nanos() as u64;
        let last = self.inner.last_adjust_ns.load(Ordering::Relaxed);
        if last != 0 && now_ns.saturating_sub(last) < interval_ns {
            return;
        }
        if self
            .inner
            .last_adjust_ns
            .compare_exchange(last, now_ns, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let Some(cores_used) = process_cpu_usage_cores() else {
            return;
        };
        let ratio = cores_used / self.inner.num_cpus as f32;
        let current = self.current_mode();
        let next = match current {
            SearchMode::HighIo if ratio > HIGH_CPU_THRESHOLD => SearchMode::HighCpu,
            SearchMode::HighCpu if ratio < LOW_CPU_THRESHOLD => SearchMode::HighIo,
            SearchMode::HighIo | SearchMode::HighCpu => return,
        };
        self.inner.mode.store(next.as_u8(), Ordering::Relaxed);
        log::debug!(
            "adaptive search pool: switching mode {} -> {} (cpu ratio {:.2})",
            current.as_str(),
            next.as_str(),
            ratio,
        );
    }
}

// =============================================================================
// Fallback constructors
//
// Compiled into the production binary but only hit at runtime by integration
// tests that pass `None` for the search runtime to `Collection::new` /
// `Collection::load`. Production wiring always constructs an adaptive handle
// via `AdaptiveSearchHandle::new` and threads it through, so these paths are
// unreachable in real deployments.
// =============================================================================

impl AdaptiveSearchHandle {
    /// Non-adaptive handle bound to the current runtime. Used as the
    /// `unwrap_or_else` fallback when a caller passes `None` for the search
    /// runtime.
    pub fn current() -> Self {
        Self::new_fixed(Handle::current())
    }

    /// Same as [`current`](Self::current), but only for tests.
    #[cfg(any(test, feature = "testing"))]
    pub fn current_for_tests() -> Self {
        Self::new_fixed(Handle::current())
    }
}

// =============================================================================
// Test and bench helpers
//
// Direct-construction helpers for tests and bench harnesses that want a
// predictable, non-adaptive handle without wiring up two runtimes.
// =============================================================================

impl AdaptiveSearchHandle {
    /// Non-adaptive handle bound to a single Tokio runtime. Both modes route
    /// to the same handle; `maybe_adjust` is disabled (via
    /// `last_adjust_ns = u64::MAX`) so no CPU sampling happens.
    pub fn new_fixed(handle: Handle) -> Self {
        let capacity = common::defaults::search_thread_count(common::cpu::get_num_cpus()).max(1);
        let inner = Arc::new(Inner {
            high_cpu: handle.clone(),
            high_io: handle,
            num_cpus: common::cpu::get_num_cpus().max(1),
            mode: AtomicU8::new(SearchMode::HighIo.as_u8()),
            last_adjust_ns: AtomicU64::new(u64::MAX),
            start: Instant::now(),
            high_cpu_session_slots: Arc::new(Semaphore::new(capacity)),
            high_io_session_slots: Arc::new(Semaphore::new(capacity)),
            high_cpu_session_capacity: capacity,
            high_io_session_capacity: capacity,
        });
        Self { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_blocking_runs_closure() {
        let adaptive = AdaptiveSearchHandle::new_fixed(Handle::current());
        let jh = adaptive.spawn_blocking(|| 42u32);
        assert_eq!(jh.await.unwrap(), 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fixed_handle_runs_many_tasks() {
        let adaptive = AdaptiveSearchHandle::new_fixed(Handle::current());
        let mut joins = Vec::new();
        for i in 0..16u32 {
            joins.push(adaptive.spawn_blocking(move || i * 2));
        }
        let mut sum = 0u32;
        for jh in joins {
            sum += jh.await.unwrap();
        }
        let expected: u32 = (0..16u32).map(|i| i * 2).sum();
        assert_eq!(sum, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exact_session_reservation_is_all_or_nothing() {
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            Handle::current(),
            Handle::current(),
            2,
            2,
        );
        let reservation = adaptive
            .try_reserve_exact_workers(1)
            .expect("entire exact session fits");
        assert_eq!(reservation.worker_slots(), 1);
        assert!(adaptive.try_reserve_exact_workers(1).is_none());
        drop(reservation);
        assert!(adaptive.try_reserve_exact_workers(1).is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oversized_exact_session_never_partially_reserves() {
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            Handle::current(),
            Handle::current(),
            3,
            3,
        );
        assert!(adaptive.try_reserve_exact_workers(3).is_none());
        let full = adaptive
            .try_reserve_exact_workers(2)
            .expect("failed oversized attempt consumed no slots");
        assert_eq!(full.worker_slots(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exact_session_cannot_spawn_beyond_reserved_task_set() {
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            Handle::current(),
            Handle::current(),
            2,
            2,
        );
        let reservation = adaptive.try_reserve_exact_workers(1).unwrap();
        let first = reservation.workers().spawn_blocking(|| 7).unwrap();
        let second = reservation.coordinator().spawn_blocking(|| 9).unwrap();
        assert!(reservation.workers().spawn_blocking(|| 11).is_none());
        assert_eq!(first.await.unwrap(), 7);
        assert_eq!(second.await.unwrap(), 9);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_join_handle_does_not_release_running_task_capacity() {
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            Handle::current(),
            Handle::current(),
            2,
            2,
        );
        let reservation = adaptive.try_reserve_exact_workers(1).unwrap();
        let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task_release = release.clone();
        let task = reservation
            .coordinator()
            .spawn_blocking(move || {
                while !task_release.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }
            })
            .unwrap();
        drop(task);
        drop(reservation);
        assert!(adaptive.try_reserve_exact_workers(1).is_none());
        release.store(true, Ordering::Relaxed);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if adaptive.try_reserve_exact_workers(1).is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("capacity must be released after the detached task exits");
    }

    #[test]
    fn exact_workers_use_the_other_qdrant_runtime_for_the_coordinator() {
        let high_cpu = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let high_io = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            high_cpu.handle().clone(),
            high_io.handle().clone(),
            2,
            2,
        );

        let reservation = adaptive
            .try_reserve_exact_workers(2)
            .expect("two cursor workers and one coordinator fit across distinct runtimes");
        assert_eq!(reservation.worker_slots(), 2);
        assert!(reservation.workers().spawn_blocking(|| 7).is_some());
        assert!(reservation.workers().spawn_blocking(|| 9).is_some());
        assert!(reservation.workers().spawn_blocking(|| 11).is_none());
        assert!(reservation.coordinator().spawn_blocking(|| 13).is_some());
        assert!(reservation.coordinator().spawn_blocking(|| 15).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exact_workers_do_not_fake_split_capacity_on_one_runtime() {
        let adaptive = AdaptiveSearchHandle::new_with_session_capacities(
            Handle::current(),
            Handle::current(),
            2,
            2,
        );
        assert!(adaptive.try_reserve_exact_workers(2).is_none());
    }
}
