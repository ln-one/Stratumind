// Copyright 2026 Stratumind contributors.
// Licensed under the Apache License, Version 2.0.

//! Production Exact-RRF HTTP contract and local execution boundary.

use std::collections::HashMap;

use actix_web::{Responder, post, web};
use actix_web_validator::{Json, Path, Query};
use collection::collection::exact_rrf::{
    DEFAULT_EXACT_BATCH_SIZE, DEFAULT_SPARSE_POSTING_BATCH_SIZE, ExactRrfRequest, ExactRrfService,
};
use collection::operations::point_ops::VectorPersisted;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::universal_query::collection_query::{
    CollectionQueryRequest, Query as CollectionQuery, VectorInputInternal, VectorQuery,
};
use segment::common::reciprocal_rank_fusion::DynamicRrfStopReason;
use segment::data_types::vectors::VectorInternal;
use segment::index::dense_rank_state::DenseExecutionPolicy;
use segment::types::{
    ExtendedPointId, Filter, SearchParams, VectorNameBuf, WithPayloadInterface, WithVector,
};
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::SparseVector;
use storage::content_manager::collection_verification::check_strict_mode_batch;
use storage::content_manager::errors::StorageError;
use storage::dispatcher::Dispatcher;
use storage::rbac::AccessRequirements;
use tokio::time::Instant;

use super::super::CollectionPath;
use super::super::read_params::ReadParams;
use crate::actix::auth::ActixAuth;
use crate::actix::helpers::{self, get_request_hardware_counter};
use crate::common::inference::bm25_inference::Bm25;
use crate::common::inference::inference_input::InferenceInput;
use crate::settings::ServiceConfig;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfDenseChannel {
    query: Vec<f32>,
    using: VectorNameBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfSparseChannel {
    query: ExactRrfSparseQuery,
    using: VectorNameBuf,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExactRrfSparseQuery {
    Vector(SparseVector),
    Document(ExactRrfBm25Document),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfBm25Document {
    text: String,
    model: String,
    #[serde(default)]
    options: Option<HashMap<String, serde_json::Value>>,
}

const EXACT_RRF_BM25_OPTION_KEYS: &[&str] = &[
    "k",
    "b",
    "avg_len",
    "tokenizer",
    "language",
    "lowercase",
    "ascii_folding",
    "stopwords",
    "stemmer",
    "min_token_len",
    "max_token_len",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfDefinition {
    dense: ExactRrfDenseChannel,
    sparse: ExactRrfSparseChannel,
    k: usize,
    #[serde(default = "default_exact_rrf_weights")]
    weights: [f32; 2],
}

fn default_exact_rrf_weights() -> [f32; 2] {
    [1.0, 1.0]
}

#[derive(Debug, Deserialize, validator::Validate)]
#[serde(deny_unknown_fields)]
struct ExactRrfQueryRequest {
    exact_rrf: ExactRrfDefinition,
    limit: usize,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    shard_key: Option<api::rest::ShardKeySelector>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactRrfHit {
    id: ExtendedPointId,
    rank: usize,
    version: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactRrfGuarantee {
    scope: &'static str,
    ordered_top_k_exact: bool,
    tie_break: &'static str,
    channel_input: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactRrfExecutionResponse {
    plan: &'static str,
    stop_reason: &'static str,
    source_pulls: Vec<usize>,
    source_exhausted: Vec<bool>,
    certification_checks: usize,
    corpus_points_observed: usize,
    query_rounds: usize,
    source_points_materialized: Vec<usize>,
    exhaustive_fallback: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExactRrfQueryResponse {
    points: Vec<ExactRrfHit>,
    guarantee: ExactRrfGuarantee,
    execution: ExactRrfExecutionResponse,
}

fn exact_channel_request(
    vector: VectorInternal,
    using: VectorNameBuf,
    filter: Option<Filter>,
    limit: usize,
) -> CollectionQueryRequest {
    CollectionQueryRequest {
        prefetch: Vec::new(),
        query: Some(CollectionQuery::Vector(VectorQuery::Nearest(
            VectorInputInternal::Vector(vector),
        ))),
        using,
        filter,
        score_threshold: None,
        limit,
        offset: 0,
        params: Some(SearchParams {
            exact: true,
            ..Default::default()
        }),
        with_vector: WithVector::Bool(false),
        with_payload: WithPayloadInterface::Bool(false),
        lookup_from: None,
    }
}

fn validate_exact_rrf_request(request: &ExactRrfQueryRequest) -> Result<(), StorageError> {
    if request.limit == 0 {
        return Err(StorageError::bad_input("exact_rrf limit must be positive"));
    }
    if request.exact_rrf.k == 0 {
        return Err(StorageError::bad_input("exact_rrf k must be positive"));
    }
    if request.exact_rrf.dense.query.is_empty()
        || request
            .exact_rrf
            .dense
            .query
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(StorageError::bad_input(
            "exact_rrf Dense query must be non-empty and finite",
        ));
    }
    match &request.exact_rrf.sparse.query {
        ExactRrfSparseQuery::Vector(sparse) => {
            if sparse.indices.len() != sparse.values.len()
                || sparse.indices.windows(2).any(|pair| pair[0] >= pair[1])
                || sparse
                    .values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(StorageError::bad_input(
                    "exact_rrf Sparse query requires sorted unique indices and finite non-negative impacts",
                ));
            }
        }
        ExactRrfSparseQuery::Document(document) => {
            if document.model != "qdrant/bm25" {
                return Err(StorageError::bad_input(
                    "exact_rrf Sparse document supports only qdrant/bm25",
                ));
            }
            if document.options.as_ref().is_some_and(|options| {
                options
                    .keys()
                    .any(|key| !EXACT_RRF_BM25_OPTION_KEYS.contains(&key.as_str()))
            }) {
                return Err(StorageError::bad_input(
                    "exact_rrf Sparse document contains an unknown BM25 option",
                ));
            }
        }
    }
    if request
        .exact_rrf
        .weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || !request.exact_rrf.weights.iter().any(|weight| *weight > 0.0)
    {
        return Err(StorageError::bad_input(
            "exact_rrf weights must be finite and non-negative with one positive value",
        ));
    }
    Ok(())
}

fn resolve_exact_sparse_query(query: ExactRrfSparseQuery) -> Result<SparseVector, StorageError> {
    match query {
        ExactRrfSparseQuery::Vector(vector) => Ok(vector),
        ExactRrfSparseQuery::Document(document) => {
            let config = InferenceInput::parse_bm25_config(document.options)?;
            match Bm25::new(config)?.search_embed(&document.text) {
                VectorPersisted::Sparse(vector) => Ok(vector),
                VectorPersisted::Dense(_) | VectorPersisted::MultiDense(_) => {
                    Err(StorageError::service_error(
                        "exact_rrf local BM25 inference returned a non-Sparse vector",
                    ))
                }
            }
        }
    }
}

fn require_default_consistency(explicit_consistency: bool) -> Result<(), StorageError> {
    if explicit_consistency {
        return Err(StorageError::bad_input(
            "exact_rrf explicit replica consistency is unsupported; use the frozen local Shard view",
        ));
    }
    Ok(())
}

async fn execute_exact_rrf_request(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<ExactRrfQueryRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
) -> impl Responder {
    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();

    let result = async {
        let request = request.into_inner();
        validate_exact_rrf_request(&request)?;
        require_default_consistency(params.consistency.is_some())?;

        let shard_selection = match request.shard_key {
            None => ShardSelectorInternal::All,
            Some(shard_keys) => shard_keys.into(),
        };
        let ExactRrfDefinition {
            dense,
            sparse,
            k: rrf_k,
            weights,
        } = request.exact_rrf;
        let sparse_query = resolve_exact_sparse_query(sparse.query)?;
        let channel_requests = [
            exact_channel_request(
                VectorInternal::Dense(dense.query.clone()),
                dense.using.clone(),
                request.filter.clone(),
                request.limit,
            ),
            exact_channel_request(
                VectorInternal::Sparse(sparse_query.clone()),
                sparse.using.clone(),
                request.filter.clone(),
                request.limit,
            ),
        ];
        let pass = check_strict_mode_batch(
            channel_requests.iter(),
            params.timeout_as_secs(),
            Some(channel_requests.len()),
            &collection.collection_name,
            &dispatcher,
            &auth,
        )
        .await?;

        let collection_pass = auth.check_collection_access(
            &collection.collection_name,
            AccessRequirements::new(),
            "query_points_exact_rrf_local",
        )?;
        let collection_ref = dispatcher
            .toc(&auth, &pass)
            .get_collection(&collection_pass)
            .await?;
        let execution = ExactRrfService::new(&collection_ref)
            .execute(
                ExactRrfRequest {
                    dense_query: dense.query,
                    dense_using: dense.using,
                    sparse_query,
                    sparse_using: sparse.using,
                    filter: request.filter,
                    limit: request.limit,
                    rrf_k,
                    weights,
                    batch_size: DEFAULT_EXACT_BATCH_SIZE,
                    sparse_posting_batch_size: DEFAULT_SPARSE_POSTING_BATCH_SIZE,
                    dense_policy: DenseExecutionPolicy::default(),
                },
                &shard_selection,
                params.timeout(),
            )
            .await?;

        let stop_reason = match execution.stop_reason {
            DynamicRrfStopReason::TopKFixed => "top-k-fixed",
            DynamicRrfStopReason::AllSourcesExhausted => "all-sources-exhausted",
        };
        let points = execution
            .point_ids
            .iter()
            .enumerate()
            .map(|(rank, &id)| {
                let version = execution.versions.get(&id).copied().ok_or_else(|| {
                    StorageError::service_error(format!(
                        "exact_rrf lost the authoritative version for point {id}"
                    ))
                })?;
                Ok(ExactRrfHit {
                    id,
                    rank: rank + 1,
                    version,
                })
            })
            .collect::<Result<Vec<_>, StorageError>>()?;

        Ok(ExactRrfQueryResponse {
            points,
            guarantee: ExactRrfGuarantee {
                scope: "selected-local-shards-frozen-segment-view",
                ordered_top_k_exact: true,
                tie_break: "point-identity-ascending",
                channel_input: "exact-channel-rank-streams",
            },
            execution: ExactRrfExecutionResponse {
                plan: "exact-rank-session-v1",
                stop_reason,
                source_pulls: execution.source_pulls,
                source_exhausted: execution.source_exhausted,
                certification_checks: execution.certification_checks,
                corpus_points_observed: execution.visible_point_copies,
                query_rounds: 1,
                source_points_materialized: execution.source_points_materialized,
                exhaustive_fallback: execution.exhaustive_fallback_sources > 0,
            },
        })
    }
    .await;

    helpers::process_response(result, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/query/exact-rrf")]
pub(super) async fn query_points_exact_rrf(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<ExactRrfQueryRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    auth: ActixAuth,
) -> impl Responder {
    execute_exact_rrf_request(
        dispatcher,
        collection,
        request,
        params,
        service_config,
        auth,
    )
    .await
}

#[cfg(test)]
include!("exact_rrf/tests.rs");
