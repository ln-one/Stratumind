// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

use std::collections::HashSet;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use actix_web::{App, HttpResponse, HttpServer, Responder, get, post, web};
use fs_err::File;
use segment::common::reciprocal_rank_fusion::{DynamicRrfPolicy, DynamicRrfStopReason};
use segment::index::dense_ball::DenseDocument;
use segment::index::n_channel_exact::{
    ExactChannelQuery, NChannelExactError, NChannelExactIndex, SparseStreamStrategy,
};
use segment::types::ExtendedPointId;
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::RemappedSparseVector;
use sparse::index::block_max::SparseDocument;

const DEFAULT_BIND: &str = "127.0.0.1:6334";
const DEFAULT_BLOCK_SIZE: usize = 128;
const MAX_TOP_K: usize = 10_000;
const MAX_CHANNELS: usize = 64;

#[derive(Debug, Deserialize)]
struct CorpusRow {
    ordinal: u32,
    source_id: String,
    dense: Vec<f32>,
    sparse_indices: Vec<u32>,
    sparse_values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SnapshotManifest {
    outputs: SnapshotOutputs,
}

#[derive(Debug, Deserialize)]
struct SnapshotOutputs {
    corpus: String,
}

struct ServerState {
    index: Arc<NChannelExactIndex>,
    source_ids: Arc<Vec<String>>,
    snapshot: String,
    documents: usize,
    dense_dimension: usize,
    build_ns: u128,
}

struct CancellationOnDrop {
    stopped: Arc<AtomicBool>,
    armed: bool,
}

impl CancellationOnDrop {
    fn new(stopped: Arc<AtomicBool>) -> Self {
        Self {
            stopped,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancellationOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.stopped.store(true, Relaxed);
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum ExactChannelRequest {
    DenseF32 {
        channel_id: String,
        vector: Vec<f32>,
        weight: f32,
    },
    SparseImpact {
        channel_id: String,
        indices: Vec<u32>,
        values: Vec<f32>,
        weight: f32,
    },
}

impl ExactChannelRequest {
    fn channel_id(&self) -> &str {
        match self {
            Self::DenseF32 { channel_id, .. } | Self::SparseImpact { channel_id, .. } => channel_id,
        }
    }

    fn weight(&self) -> f32 {
        match self {
            Self::DenseF32 { weight, .. } | Self::SparseImpact { weight, .. } => *weight,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactQueryRequest {
    index_generation_id: String,
    channels: Vec<ExactChannelRequest>,
    rerank_candidate_k: usize,
    rrf_k: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactQueryResponse {
    index_generation_id: String,
    hits: Vec<ExactHit>,
    guarantee: ExactGuarantee,
    channels: Vec<ExactChannelProgress>,
    stop: ExactStop,
    physical: ExactPhysical,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactHit {
    point_id: String,
    ordinal: u64,
    rank: usize,
    fusion_score: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactGuarantee {
    scope: &'static str,
    ordered_top_k_exact: bool,
    scores_exact: bool,
    tie_break: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactStop {
    reason: &'static str,
    source_pulls: Vec<usize>,
    certification_checks: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactChannelProgress {
    channel_id: String,
    logical_rank_pulls: usize,
    exhausted: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactPhysical {
    executor: &'static str,
    physical_streams: usize,
    physical_rank_pulls: usize,
    peak_replay_buffered_identities: usize,
    dense_logical_dot_products: usize,
    dense_physical_dot_products: usize,
    duration_micros: u128,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    status: &'static str,
    mode: &'static str,
    snapshot: String,
    documents: usize,
    dense_dimension: usize,
    build_millis: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[get("/health")]
async fn health(state: web::Data<ServerState>) -> impl Responder {
    web::Json(HealthResponse {
        status: "ok",
        mode: "reference-exact",
        snapshot: state.snapshot.clone(),
        documents: state.documents,
        dense_dimension: state.dense_dimension,
        build_millis: state.build_ns as f64 / 1_000_000.0,
    })
}

#[post("/spectra/query")]
async fn exact_query(
    state: web::Data<ServerState>,
    request: web::Json<ExactQueryRequest>,
) -> impl Responder {
    if request.index_generation_id != state.snapshot {
        return bad_request("indexGenerationId does not match the loaded snapshot");
    }
    if request.rerank_candidate_k == 0 || request.rerank_candidate_k > MAX_TOP_K {
        return bad_request("rerankCandidateK must be between 1 and 10000");
    }
    if request.rerank_candidate_k > state.documents {
        return bad_request("rerankCandidateK exceeds the snapshot document count");
    }
    if request.rrf_k == 0 {
        return bad_request("rrfK must be positive");
    }
    let weights = match validate_channels(&request.channels, state.dense_dimension) {
        Ok(weights) => weights,
        Err(error) => return bad_request(error),
    };
    let channel_ids = request
        .channels
        .iter()
        .map(|channel| channel.channel_id().to_owned())
        .collect::<Vec<_>>();

    let index = Arc::clone(&state.index);
    let channels = request.channels.clone();
    let top_k = request.rerank_candidate_k;
    let rrf_k = request.rrf_k;
    let started = Instant::now();
    let stopped = Arc::new(AtomicBool::new(false));
    let mut cancellation_on_drop = CancellationOnDrop::new(Arc::clone(&stopped));
    let worker_stopped = Arc::clone(&stopped);
    let execution = web::block(move || {
        let queries = channels
            .iter()
            .map(|channel| match channel {
                ExactChannelRequest::DenseF32 { vector, .. } => ExactChannelQuery::Dense(vector),
                ExactChannelRequest::SparseImpact {
                    indices, values, ..
                } => ExactChannelQuery::Sparse(RemappedSparseVector {
                    indices: indices.clone(),
                    values: values.clone(),
                }),
            })
            .collect();
        index.search_with_sparse_strategy_and_cancellation(
            queries,
            top_k,
            rrf_k,
            Some(&weights),
            DynamicRrfPolicy::default(),
            SparseStreamStrategy::default(),
            worker_stopped.as_ref(),
        )
    })
    .await;
    cancellation_on_drop.disarm();
    let duration_micros = started.elapsed().as_micros();

    let result = match execution {
        Ok(Ok(result)) => result,
        Ok(Err(NChannelExactError::Cancelled)) => {
            return HttpResponse::RequestTimeout().json(ErrorResponse {
                error: "exact retrieval was cancelled".to_owned(),
            });
        }
        Ok(Err(error)) => return bad_request(error.to_string()),
        Err(error) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("exact retrieval worker failed: {error}"),
            });
        }
    };
    if result.execution.source_pulls.len() != channel_ids.len()
        || result.execution.source_exhausted.len() != channel_ids.len()
    {
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: "exact executor returned invalid per-channel progress".to_owned(),
        });
    }
    let mut hits = Vec::with_capacity(result.execution.point_ids.len());
    for (position, point_id) in result.execution.point_ids.iter().enumerate() {
        let ExtendedPointId::NumId(ordinal) = point_id else {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: "exact snapshot returned a non-numeric internal identity".to_owned(),
            });
        };
        let Some(source_id) = state.source_ids.get(*ordinal as usize) else {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: "exact snapshot returned an out-of-range identity".to_owned(),
            });
        };
        hits.push(ExactHit {
            point_id: source_id.clone(),
            ordinal: *ordinal,
            rank: position + 1,
            fusion_score: None,
        });
    }

    let stop_reason = match result.execution.stop_reason {
        DynamicRrfStopReason::TopKFixed => "top-k-fixed",
        DynamicRrfStopReason::AllSourcesExhausted => "all-sources-exhausted",
    };
    HttpResponse::Ok().json(ExactQueryResponse {
        index_generation_id: state.snapshot.clone(),
        hits,
        guarantee: ExactGuarantee {
            scope: "full-corpus",
            ordered_top_k_exact: true,
            scores_exact: false,
            tie_break: "frozen-point-identity-ascending",
        },
        channels: channel_ids
            .into_iter()
            .zip(&result.execution.source_pulls)
            .zip(&result.execution.source_exhausted)
            .map(
                |((channel_id, &logical_rank_pulls), &exhausted)| ExactChannelProgress {
                    channel_id,
                    logical_rank_pulls,
                    exhausted,
                },
            )
            .collect(),
        stop: ExactStop {
            reason: stop_reason,
            source_pulls: result.execution.source_pulls,
            certification_checks: result.execution.certification_checks,
        },
        physical: ExactPhysical {
            executor: "n-channel-exact",
            physical_streams: result.physical.physical_rank_streams,
            physical_rank_pulls: result.physical.physical_rank_pulls,
            peak_replay_buffered_identities: result.physical.rank_replay_buffered_identities,
            dense_logical_dot_products: result.physical.dense_logical_dot_products,
            dense_physical_dot_products: result.physical.dense_physical_dot_products,
            duration_micros,
        },
    })
}

fn validate_channels(
    channels: &[ExactChannelRequest],
    dense_dimension: usize,
) -> Result<Vec<f32>, String> {
    if channels.is_empty() || channels.len() > MAX_CHANNELS {
        return Err(format!(
            "channels must contain between 1 and {MAX_CHANNELS} entries"
        ));
    }
    let mut channel_ids = HashSet::with_capacity(channels.len());
    let mut weights = Vec::with_capacity(channels.len());
    for channel in channels {
        if channel.channel_id().trim().is_empty() {
            return Err("channelId must not be empty".to_owned());
        }
        if !channel_ids.insert(channel.channel_id()) {
            return Err(format!("duplicate channelId: {}", channel.channel_id()));
        }
        let weight = channel.weight();
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "channel {} weight must be finite and non-negative",
                channel.channel_id()
            ));
        }
        weights.push(weight);
        match channel {
            ExactChannelRequest::DenseF32 { vector, .. } => {
                if vector.len() != dense_dimension {
                    return Err(format!(
                        "channel {} Dense dimension does not match the snapshot",
                        channel.channel_id()
                    ));
                }
                if vector.iter().any(|value| !value.is_finite()) {
                    return Err(format!(
                        "channel {} Dense values must be finite",
                        channel.channel_id()
                    ));
                }
            }
            ExactChannelRequest::SparseImpact {
                indices, values, ..
            } => {
                if indices.len() != values.len() {
                    return Err(format!(
                        "channel {} Sparse indices and values must have equal length",
                        channel.channel_id()
                    ));
                }
                if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return Err(format!(
                        "channel {} Sparse indices must be sorted and unique",
                        channel.channel_id()
                    ));
                }
                if values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(format!(
                        "channel {} Sparse values must be finite and non-negative",
                        channel.channel_id()
                    ));
                }
            }
        }
    }
    if !weights.iter().any(|weight| *weight > 0.0) {
        return Err("at least one channel weight must be positive".to_owned());
    }
    Ok(weights)
}

fn bad_request(message: impl Into<String>) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorResponse {
        error: message.into(),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let dataset_dir = PathBuf::from(
        env::var("SPECTRA_DATASET_DIR")
            .expect("SPECTRA_DATASET_DIR must point to a prepared vector snapshot"),
    );
    let bind = env::var("SPECTRA_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_owned());
    let block_size = env::var("SPECTRA_BLOCK_SIZE")
        .ok()
        .map(|value| value.parse().expect("invalid SPECTRA_BLOCK_SIZE"))
        .unwrap_or(DEFAULT_BLOCK_SIZE);
    assert!(block_size > 0, "SPECTRA_BLOCK_SIZE must be positive");

    let manifest: SnapshotManifest = read_json(&dataset_dir.join("manifest.json"));
    let rows: Vec<CorpusRow> = read_jsonl(&dataset_dir.join("corpus-vectors.jsonl"));
    assert!(!rows.is_empty(), "snapshot corpus must not be empty");
    let dense_dimension = rows[0].dense.len();
    let mut source_ids = Vec::with_capacity(rows.len());
    let mut sparse_documents = Vec::with_capacity(rows.len());
    let mut dense_documents = Vec::with_capacity(rows.len());
    for (position, row) in rows.into_iter().enumerate() {
        assert_eq!(
            row.ordinal as usize, position,
            "ordinals must be contiguous"
        );
        assert_eq!(row.dense.len(), dense_dimension, "Dense dimension mismatch");
        source_ids.push(row.source_id);
        sparse_documents.push(SparseDocument {
            id: row.ordinal,
            vector: RemappedSparseVector {
                indices: row.sparse_indices,
                values: row.sparse_values,
            },
        });
        dense_documents.push(DenseDocument {
            id: row.ordinal,
            vector: row.dense,
        });
    }

    let build_started = Instant::now();
    let index = NChannelExactIndex::build(dense_documents, sparse_documents, block_size)
        .expect("failed to build exact snapshot index");
    let state = web::Data::new(ServerState {
        documents: index.document_count(),
        index: Arc::new(index),
        source_ids: Arc::new(source_ids),
        snapshot: manifest.outputs.corpus,
        dense_dimension,
        build_ns: build_started.elapsed().as_nanos(),
    });

    eprintln!(
        "Spectra exact server listening on {bind} with {} documents at snapshot {}",
        state.documents, state.snapshot
    );
    HttpServer::new(move || {
        App::new()
            .app_data(web::JsonConfig::default().limit(1 << 20))
            .app_data(state.clone())
            .service(health)
            .service(exact_query)
    })
    .h1_allow_half_closed(false)
    .bind(bind)?
    .run()
    .await
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_reader(BufReader::new(File::open(path).unwrap()))
        .unwrap_or_else(|error| panic!("invalid {}: {error}", path.display()))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    BufReader::new(File::open(path).unwrap())
        .lines()
        .enumerate()
        .filter_map(|(position, line)| {
            let line = line.unwrap();
            (!line.trim().is_empty()).then(|| {
                serde_json::from_str(&line).unwrap_or_else(|error| {
                    panic!("invalid {}:{}: {error}", path.display(), position + 1)
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::time::Duration;

    use super::*;

    struct DisconnectTestState {
        handler_started: Arc<AtomicBool>,
        stopped: Arc<AtomicBool>,
        worker_observed_cancellation: Arc<AtomicBool>,
    }

    async fn cancellable_worker(state: web::Data<DisconnectTestState>) -> HttpResponse {
        let mut guard = CancellationOnDrop::new(Arc::clone(&state.stopped));
        let worker_stopped = Arc::clone(&state.stopped);
        let worker_observed_cancellation = Arc::clone(&state.worker_observed_cancellation);
        state.handler_started.store(true, Relaxed);
        let _ = web::block(move || {
            while !worker_stopped.load(Relaxed) {
                std::thread::yield_now();
            }
            worker_observed_cancellation.store(true, Relaxed);
        })
        .await;
        guard.disarm();
        HttpResponse::Ok().finish()
    }

    async fn wait_until_true(flag: &AtomicBool) -> bool {
        for _ in 0..200 {
            if flag.load(Relaxed) {
                return true;
            }
            actix_web::rt::time::sleep(Duration::from_millis(5)).await;
        }
        false
    }

    #[test]
    fn armed_cancellation_guard_sets_the_worker_flag_on_drop() {
        let stopped = Arc::new(AtomicBool::new(false));

        drop(CancellationOnDrop::new(Arc::clone(&stopped)));

        assert!(stopped.load(Relaxed));
    }

    #[test]
    fn disarmed_cancellation_guard_leaves_the_worker_running() {
        let stopped = Arc::new(AtomicBool::new(false));
        let mut guard = CancellationOnDrop::new(Arc::clone(&stopped));
        guard.disarm();

        drop(guard);

        assert!(!stopped.load(Relaxed));
    }

    #[actix_web::test]
    async fn http1_disconnect_cancels_an_in_flight_blocking_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handler_started = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_observed_cancellation = Arc::new(AtomicBool::new(false));
        let state = web::Data::new(DisconnectTestState {
            handler_started: Arc::clone(&handler_started),
            stopped: Arc::clone(&stopped),
            worker_observed_cancellation: Arc::clone(&worker_observed_cancellation),
        });
        let server = HttpServer::new(move || {
            App::new()
                .app_data(state.clone())
                .route("/slow", web::post().to(cancellable_worker))
        })
        .h1_allow_half_closed(false)
        .listen(listener)
        .unwrap()
        .run();
        let server_handle = server.handle();
        actix_web::rt::spawn(server);

        let mut client = TcpStream::connect(address).unwrap();
        client
            .write_all(
                b"POST /slow HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            )
            .unwrap();
        assert!(wait_until_true(&handler_started).await);

        client.shutdown(Shutdown::Both).unwrap();

        assert!(wait_until_true(&stopped).await);
        assert!(wait_until_true(&worker_observed_cancellation).await);
        server_handle.stop(true).await;
    }

    #[test]
    fn empty_sparse_channel_is_valid() {
        let channels = vec![ExactChannelRequest::SparseImpact {
            channel_id: "sparse-empty".to_owned(),
            indices: Vec::new(),
            values: Vec::new(),
            weight: 1.0,
        }];
        assert_eq!(validate_channels(&channels, 2).unwrap(), vec![1.0]);
    }

    #[test]
    fn n_channel_wire_shape_deserializes() {
        let request: ExactQueryRequest = serde_json::from_value(serde_json::json!({
            "indexGenerationId": "snapshot",
            "channels": [
                {
                    "channelId": "dense",
                    "kind": "dense-f32",
                    "vector": [1.0, 0.0],
                    "weight": 1.0
                },
                {
                    "channelId": "sparse",
                    "kind": "sparse-impact",
                    "indices": [],
                    "values": [],
                    "weight": 0.5
                }
            ],
            "rerankCandidateK": 20,
            "rrfK": 60
        }))
        .unwrap();

        assert_eq!(request.channels.len(), 2);
        assert_eq!(request.rerank_candidate_k, 20);
    }

    #[test]
    fn duplicate_channel_identity_is_rejected() {
        let channels = vec![
            ExactChannelRequest::DenseF32 {
                channel_id: "same".to_owned(),
                vector: vec![1.0, 0.0],
                weight: 1.0,
            },
            ExactChannelRequest::SparseImpact {
                channel_id: "same".to_owned(),
                indices: vec![1],
                values: vec![1.0],
                weight: 1.0,
            },
        ];
        assert!(validate_channels(&channels, 2).is_err());
    }

    #[test]
    fn invalid_dense_sparse_and_weight_inputs_are_rejected() {
        let cases = [
            ExactChannelRequest::DenseF32 {
                channel_id: "dense-dimension".to_owned(),
                vector: vec![1.0],
                weight: 1.0,
            },
            ExactChannelRequest::DenseF32 {
                channel_id: "dense-nan".to_owned(),
                vector: vec![f32::NAN, 0.0],
                weight: 1.0,
            },
            ExactChannelRequest::SparseImpact {
                channel_id: "sparse-order".to_owned(),
                indices: vec![2, 1],
                values: vec![1.0, 1.0],
                weight: 1.0,
            },
            ExactChannelRequest::SparseImpact {
                channel_id: "negative-weight".to_owned(),
                indices: vec![1],
                values: vec![1.0],
                weight: -1.0,
            },
        ];
        for channel in cases {
            assert!(validate_channels(&[channel], 2).is_err());
        }
    }
}
