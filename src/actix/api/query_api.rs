use actix_web::{Responder, post, web};
use actix_web_validator::{Json, Path, Query};
use api::rest::models::InferenceUsage;
use api::rest::{QueryGroupsRequest, QueryRequest, QueryRequestBatch, QueryResponse};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::CountRequestInternal;
use collection::operations::universal_query::collection_query::{
    CollectionQueryRequest, Query as CollectionQuery, VectorInputInternal, VectorQuery,
};
use itertools::Itertools;
use segment::common::reciprocal_rank_fusion::{
    DynamicRrfAdvance, DynamicRrfPolicy, DynamicRrfScheduler, DynamicRrfSession,
    DynamicRrfStopReason, ExactRrfStream, infallible_exact_rrf_stream,
};
use segment::data_types::vectors::VectorInternal;
use segment::types::{
    ExtendedPointId, Filter, SearchParams, VectorNameBuf, WithPayloadInterface, WithVector,
};
use serde::{Deserialize, Serialize};
use sparse::common::sparse_vector::SparseVector;
use storage::content_manager::collection_verification::{
    check_strict_mode, check_strict_mode_batch,
};
use storage::content_manager::errors::StorageError;
use storage::dispatcher::Dispatcher;
use tokio::time::Instant;

use super::CollectionPath;
use super::read_params::ReadParams;
use crate::actix::auth::ActixAuth;
use crate::actix::helpers::{self, get_request_hardware_counter};
use crate::common::inference::api_keys::InferenceApiKeys;
use crate::common::inference::params::InferenceParams;
use crate::common::inference::query_requests_rest::{
    CollectionQueryGroupsRequestWithUsage, CollectionQueryRequestWithUsage,
    convert_query_groups_request_from_rest, convert_query_request_from_rest,
};
use crate::common::query::do_query_point_groups;
use crate::settings::ServiceConfig;

#[cfg(test)]
pub const THIS_FILE: &str = file!();

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfDenseChannel {
    query: Vec<f32>,
    using: VectorNameBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRrfSparseChannel {
    query: SparseVector,
    using: VectorNameBuf,
}

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

fn normalize_exact_channel_ties(points: &mut [segment::types::ScoredPoint]) {
    let mut start = 0;
    while start < points.len() {
        let score = points[start].score;
        let mut end = start + 1;
        while end < points.len() && points[end].score == score {
            end += 1;
        }
        points[start..end].sort_unstable_by(|left, right| left.id.cmp(&right.id));
        start = end;
    }
}

/// Returns the prefix whose final score group is known to be complete.
///
/// Qdrant's native Top-N is score exact, but an arbitrary equal-score subset
/// may straddle N. One probe point detects that boundary. The incomplete tie
/// group is withheld until a deeper round closes it or reaches exact EOF.
fn tie_complete_prefix_len(
    points: &[segment::types::ScoredPoint],
    requested_prefix: usize,
    corpus_points: usize,
) -> (usize, bool) {
    let query_limit = requested_prefix.saturating_add(1).min(corpus_points);
    let complete = query_limit == corpus_points || points.len() < query_limit;
    if complete {
        return (points.len(), true);
    }
    debug_assert!(points.len() > requested_prefix);
    let boundary_score = points[requested_prefix].score;
    let first_boundary = points[..=requested_prefix]
        .iter()
        .rposition(|point| point.score != boundary_score)
        .map_or(0, |position| position + 1);
    (first_boundary, false)
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
    let sparse = &request.exact_rrf.sparse.query;
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

#[post("/collections/{collection_name}/points/query")]
async fn query_points(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<QueryRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
    api_keys: InferenceApiKeys,
) -> impl Responder {
    let QueryRequest {
        internal: query_request,
        shard_key,
    } = request.into_inner();

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();

    let shard_selection = match shard_key {
        None => ShardSelectorInternal::All,
        Some(shard_keys) => shard_keys.into(),
    };
    let hw_measurement_acc = request_hw_counter.get_counter();
    let mut inference_usage = InferenceUsage::default();

    let inference_params = InferenceParams::new(api_keys, params.timeout());

    let result = async {
        let CollectionQueryRequestWithUsage { request, usage } =
            convert_query_request_from_rest(query_request, &inference_params).await?;

        inference_usage.merge_opt(usage);

        let pass = check_strict_mode(
            &request,
            params.timeout_as_secs(),
            &collection.collection_name,
            &dispatcher,
            &auth,
        )
        .await?;

        let points = dispatcher
            .toc(&auth, &pass)
            .query_batch(
                &collection.collection_name,
                vec![(request, shard_selection)],
                params.consistency,
                auth,
                params.timeout(),
                hw_measurement_acc,
            )
            .await?
            .pop()
            .ok_or_else(|| {
                StorageError::service_error("Expected at least one response for one query")
            })?
            .into_iter()
            .map(api::rest::ScoredPoint::from)
            .collect_vec();

        Ok(QueryResponse { points })
    }
    .await;

    helpers::process_response_with_inference_usage(
        result,
        timing,
        request_hw_counter.to_rest_api(),
        inference_usage.into_non_empty(),
    )
}

/// Stratumind V0 exact hybrid endpoint.
///
/// The production-safe V0 plan repeatedly asks Qdrant for exact, tie-complete
/// channel prefixes and deepens them until dynamic WRRF certifies the final
/// Top-K. It falls back to exact EOF rather than weakening the result contract.
/// Native resumable producers can later replace this repeated-query physical
/// plan without changing the request or guarantee contract.
#[post("/collections/{collection_name}/points/query/exact-rrf")]
async fn query_points_exact_rrf(
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
    let hw_measurement_acc = request_hw_counter.get_counter();

    let result = async {
        let request = request.into_inner();
        validate_exact_rrf_request(&request)?;
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
        let channel_requests = vec![
            exact_channel_request(
                VectorInternal::Dense(dense.query),
                dense.using,
                request.filter.clone(),
                request.limit,
            ),
            exact_channel_request(
                VectorInternal::Sparse(sparse.query),
                sparse.using,
                request.filter.clone(),
                request.limit,
            ),
        ];

        // Validate the user-visible final limit. The internal exhaustive limit
        // is an implementation detail of this explicit exact endpoint.
        let pass = check_strict_mode_batch(
            channel_requests.iter(),
            params.timeout_as_secs(),
            Some(channel_requests.len()),
            &collection.collection_name,
            &dispatcher,
            &auth,
        )
        .await?;

        let count_request = CountRequestInternal {
            filter: request.filter,
            exact: true,
        };
        let count_before = dispatcher
            .toc(&auth, &pass)
            .count(
                &collection.collection_name,
                count_request.clone(),
                params.consistency,
                params.timeout(),
                shard_selection.clone(),
                auth.clone(),
                hw_measurement_acc.clone(),
            )
            .await?
            .count;

        if count_before == 0 {
            let count_after = dispatcher
                .toc(&auth, &pass)
                .count(
                    &collection.collection_name,
                    count_request,
                    params.consistency,
                    params.timeout(),
                    shard_selection,
                    auth,
                    hw_measurement_acc,
                )
                .await?
                .count;
            if count_after != 0 {
                return Err(StorageError::service_error(
                    "exact_rrf corpus changed during execution; retry against a frozen generation",
                ));
            }
            return Ok(ExactRrfQueryResponse {
                points: Vec::new(),
                guarantee: ExactRrfGuarantee {
                    scope: "selected-shards-request-view",
                    ordered_top_k_exact: true,
                    tie_break: "point-identity-ascending",
                    channel_input: "tie-complete-exact-prefixes",
                },
                execution: ExactRrfExecutionResponse {
                    plan: "adaptive-exact-prefix-v0",
                    stop_reason: "all-sources-exhausted",
                    source_pulls: vec![0, 0],
                    source_exhausted: vec![true, true],
                    certification_checks: 1,
                    corpus_points_observed: 0,
                    query_rounds: 0,
                    source_points_materialized: vec![0, 0],
                    exhaustive_fallback: false,
                },
            });
        }

        const INITIAL_EXACT_PREFIX: usize = 64;
        const MAX_ADAPTIVE_ROUNDS: usize = 3;
        let mut requested_prefix = request
            .limit
            .saturating_mul(4)
            .max(INITIAL_EXACT_PREFIX)
            .min(count_before);
        let mut session: Option<DynamicRrfSession<'static>> = None;
        let mut source_complete = [false; 2];
        let mut source_available = [0; 2];
        let mut source_points_materialized = vec![0; 2];
        let mut versions = std::collections::HashMap::new();
        let mut query_rounds = 0usize;
        let mut exhaustive_fallback = false;

        let execution = loop {
            query_rounds += 1;
            let requested_sources: Vec<_> = (0..channel_requests.len())
                .filter(|source| {
                    session
                        .as_ref()
                        .and_then(|session: &DynamicRrfSession<'_>| {
                            session.source_is_exhausted(*source)
                        })
                        != Some(true)
                })
                .collect();
            if requested_sources.is_empty() {
                return Err(StorageError::service_error(
                    "exact_rrf paused after every source reached exact EOF",
                ));
            }
            let query_limit = requested_prefix.saturating_add(1).min(count_before);
            let round_requests = requested_sources
                .iter()
                .map(|&source| {
                    let mut channel = channel_requests[source].clone();
                    channel.limit = query_limit;
                    (channel, shard_selection.clone())
                })
                .collect();
            let rankings = dispatcher
                .toc(&auth, &pass)
                .query_batch(
                    &collection.collection_name,
                    round_requests,
                    params.consistency,
                    auth.clone(),
                    params.timeout(),
                    hw_measurement_acc.clone(),
                )
                .await?;
            if rankings.len() != requested_sources.len() {
                return Err(StorageError::service_error(
                    "exact_rrf received an incomplete channel batch",
                ));
            }

            let mut initial_streams: Vec<Option<ExactRrfStream<'static>>> =
                std::iter::repeat_with(|| None)
                    .take(channel_requests.len())
                    .collect();
            for (source, mut ranking) in requested_sources.iter().copied().zip(rankings) {
                source_points_materialized[source] += ranking.len();
                normalize_exact_channel_ties(&mut ranking);
                let (safe_prefix, complete) =
                    tie_complete_prefix_len(&ranking, requested_prefix, count_before);
                ranking.truncate(safe_prefix);
                source_available[source] = safe_prefix;
                source_complete[source] = complete;
                for point in &ranking {
                    versions
                        .entry(point.id)
                        .and_modify(|version: &mut u64| *version = (*version).max(point.version))
                        .or_insert(point.version);
                }
                let stream = infallible_exact_rrf_stream(ranking.into_iter().map(|point| point.id));
                if let Some(session) = session.as_mut() {
                    session
                        .replace_source(source, stream)
                        .map_err(|error| StorageError::service_error(error.to_string()))?;
                } else {
                    initial_streams[source] = Some(stream);
                }
            }

            if session.is_none() {
                let streams = initial_streams
                    .into_iter()
                    .enumerate()
                    .map(|(source, stream)| {
                        stream.ok_or_else(|| {
                            StorageError::service_error(format!(
                                "exact_rrf did not initialize source {source}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, StorageError>>()?;
                session = Some(
                    DynamicRrfSession::new(
                        streams,
                        request.limit,
                        rrf_k,
                        Some(&weights),
                        DynamicRrfPolicy {
                            scheduler: DynamicRrfScheduler::MaxNextContribution,
                            ..DynamicRrfPolicy::default()
                        },
                    )
                    .map_err(|error| StorageError::service_error(error.to_string()))?,
                );
            }

            let session = session
                .as_mut()
                .expect("exact_rrf session is initialized after the first channel batch");
            let targets: Vec<_> = (0..channel_requests.len())
                .map(|source| {
                    if session.source_is_exhausted(source) == Some(true) {
                        None
                    } else {
                        Some(
                            source_available[source]
                                .saturating_add(usize::from(source_complete[source])),
                        )
                    }
                })
                .collect();
            match session
                .advance_until_each(&targets)
                .map_err(|error| StorageError::service_error(error.to_string()))?
            {
                DynamicRrfAdvance::Fixed(execution) => break execution,
                DynamicRrfAdvance::Paused => {}
            }

            let next_prefix = if query_rounds >= MAX_ADAPTIVE_ROUNDS {
                exhaustive_fallback = true;
                count_before
            } else {
                requested_prefix.saturating_mul(4).min(count_before)
            };
            if next_prefix <= requested_prefix {
                return Err(StorageError::service_error(
                    "exact_rrf could not extend an uncertified channel prefix",
                ));
            }
            requested_prefix = next_prefix;
        };

        let count_after = dispatcher
            .toc(&auth, &pass)
            .count(
                &collection.collection_name,
                count_request,
                params.consistency,
                params.timeout(),
                shard_selection,
                auth,
                hw_measurement_acc,
            )
            .await?
            .count;
        if count_before != count_after {
            return Err(StorageError::service_error(
                "exact_rrf corpus changed during execution; retry against a frozen generation",
            ));
        }

        let stop_reason = match execution.stop_reason {
            DynamicRrfStopReason::TopKFixed => "top-k-fixed",
            DynamicRrfStopReason::AllSourcesExhausted => "all-sources-exhausted",
        };
        let points = execution
            .point_ids
            .iter()
            .enumerate()
            .map(|(rank, &id)| {
                let version = versions.get(&id).copied().ok_or_else(|| {
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
                scope: "selected-shards-request-view",
                ordered_top_k_exact: true,
                tie_break: "point-identity-ascending",
                channel_input: "tie-complete-exact-prefixes",
            },
            execution: ExactRrfExecutionResponse {
                plan: if exhaustive_fallback {
                    "adaptive-exact-prefix-with-exhaustive-fallback-v0"
                } else {
                    "adaptive-exact-prefix-v0"
                },
                stop_reason,
                source_pulls: execution.source_pulls,
                source_exhausted: execution.source_exhausted,
                certification_checks: execution.certification_checks,
                corpus_points_observed: count_before,
                query_rounds,
                source_points_materialized,
                exhaustive_fallback,
            },
        })
    }
    .await;

    helpers::process_response(result, timing, request_hw_counter.to_rest_api())
}

#[post("/collections/{collection_name}/points/query/batch")]
async fn query_points_batch(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<QueryRequestBatch>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
    api_keys: InferenceApiKeys,
) -> impl Responder {
    let QueryRequestBatch { searches } = request.into_inner();

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();
    let hw_measurement_acc = request_hw_counter.get_counter();

    let mut all_usages: InferenceUsage = InferenceUsage::default();

    let inference_params = InferenceParams::new(api_keys, params.timeout());

    let result = async {
        let mut batch = Vec::with_capacity(searches.len());

        for request_item in searches {
            let QueryRequest {
                internal,
                shard_key,
            } = request_item;

            let CollectionQueryRequestWithUsage { request, usage } =
                convert_query_request_from_rest(internal, &inference_params).await?;

            all_usages.merge_opt(usage);

            let shard_selection = match shard_key {
                None => ShardSelectorInternal::All,
                Some(shard_keys) => shard_keys.into(),
            };

            batch.push((request, shard_selection));
        }

        let pass = check_strict_mode_batch(
            batch.iter().map(|i| &i.0),
            params.timeout_as_secs(),
            Some(batch.len()),
            &collection.collection_name,
            &dispatcher,
            &auth,
        )
        .await?;

        let res = dispatcher
            .toc(&auth, &pass)
            .query_batch(
                &collection.collection_name,
                batch,
                params.consistency,
                auth,
                params.timeout(),
                hw_measurement_acc,
            )
            .await?
            .into_iter()
            .map(|response| QueryResponse {
                points: response
                    .into_iter()
                    .map(api::rest::ScoredPoint::from)
                    .collect_vec(),
            })
            .collect_vec();
        Ok(res)
    }
    .await;

    helpers::process_response_with_inference_usage(
        result,
        timing,
        request_hw_counter.to_rest_api(),
        all_usages.into_non_empty(),
    )
}

#[post("/collections/{collection_name}/points/query/groups")]
async fn query_points_groups(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<QueryGroupsRequest>,
    params: Query<ReadParams>,
    service_config: web::Data<ServiceConfig>,
    ActixAuth(auth): ActixAuth,
    api_keys: InferenceApiKeys,
) -> impl Responder {
    let QueryGroupsRequest {
        search_group_request,
        shard_key,
    } = request.into_inner();

    let request_hw_counter = get_request_hardware_counter(
        &dispatcher,
        collection.collection_name.clone(),
        service_config.hardware_reporting(),
        None,
    );
    let timing = Instant::now();
    let hw_measurement_acc = request_hw_counter.get_counter();
    let mut inference_usage = InferenceUsage::default();

    let inference_params = InferenceParams::new(api_keys, params.timeout());

    let result = async {
        let shard_selection = match shard_key {
            None => ShardSelectorInternal::All,
            Some(shard_keys) => shard_keys.into(),
        };
        let CollectionQueryGroupsRequestWithUsage { request, usage } =
            convert_query_groups_request_from_rest(search_group_request, inference_params).await?;

        inference_usage.merge_opt(usage);

        let pass = check_strict_mode(
            &request,
            params.timeout_as_secs(),
            &collection.collection_name,
            &dispatcher,
            &auth,
        )
        .await?;

        let query_result = do_query_point_groups(
            dispatcher.toc(&auth, &pass),
            &collection.collection_name,
            request,
            params.consistency,
            shard_selection,
            auth,
            params.timeout(),
            hw_measurement_acc,
        )
        .await?;
        Ok(query_result)
    }
    .await;

    helpers::process_response_with_inference_usage(
        result,
        timing,
        request_hw_counter.to_rest_api(),
        inference_usage.into_non_empty(),
    )
}

pub fn config_query_api(cfg: &mut web::ServiceConfig) {
    cfg.service(query_points);
    cfg.service(query_points_exact_rrf);
    cfg.service(query_points_batch);
    cfg.service(query_points_groups);
}

#[cfg(test)]
mod exact_rrf_tests {
    use super::*;

    fn scored(id: u64, score: f32) -> segment::types::ScoredPoint {
        segment::types::ScoredPoint {
            id: id.into(),
            version: 1,
            score,
            payload: None,
            vector: None,
            shard_key: None,
            order_value: None,
        }
    }

    #[test]
    fn exact_rrf_request_accepts_the_frozen_two_channel_shape() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1, 0.2], "using": "dense" },
                "sparse": {
                    "query": { "indices": [12, 99], "values": [1.3, 0.5] },
                    "using": "sparse"
                },
                "k": 60,
                "weights": [1.0, 1.0]
            },
            "limit": 20
        }))
        .unwrap();

        validate_exact_rrf_request(&request).unwrap();
    }

    #[test]
    fn exact_rrf_rejects_negative_sparse_impacts() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1], "using": "dense" },
                "sparse": {
                    "query": { "indices": [12], "values": [-1.0] },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        }))
        .unwrap();

        assert!(validate_exact_rrf_request(&request).is_err());
    }

    #[test]
    fn tie_normalization_changes_only_equal_score_identity_order() {
        let mut points = vec![
            scored(7, 9.0),
            scored(2, 9.0),
            scored(5, 8.0),
            scored(1, 7.0),
        ];

        normalize_exact_channel_ties(&mut points);

        assert_eq!(
            points.iter().map(|point| point.id).collect::<Vec<_>>(),
            vec![2.into(), 7.into(), 5.into(), 1.into()]
        );
        assert_eq!(
            points.iter().map(|point| point.score).collect::<Vec<_>>(),
            vec![9.0, 9.0, 8.0, 7.0]
        );
    }

    #[test]
    fn exact_prefix_withholds_a_tie_group_that_crosses_the_probe_boundary() {
        let crossing = vec![scored(1, 9.0), scored(2, 8.0), scored(3, 8.0)];
        let separated = vec![scored(1, 9.0), scored(2, 8.0), scored(3, 7.0)];

        assert_eq!(tie_complete_prefix_len(&crossing, 2, 10), (1, false));
        assert_eq!(tie_complete_prefix_len(&separated, 2, 10), (2, false));
    }

    #[test]
    fn exact_prefix_keeps_the_complete_final_tie_group_at_eof() {
        let points = vec![scored(1, 9.0), scored(2, 8.0), scored(3, 8.0)];

        assert_eq!(tie_complete_prefix_len(&points, 2, 3), (3, true));
    }
}
