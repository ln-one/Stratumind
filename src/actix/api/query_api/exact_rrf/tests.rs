#[cfg(test)]
mod exact_rrf_tests {
    use super::*;

    fn sparse_vector_mut(request: &mut ExactRrfQueryRequest) -> &mut SparseVector {
        match &mut request.exact_rrf.sparse.query {
            ExactRrfSparseQuery::Vector(vector) => vector,
            ExactRrfSparseQuery::Document(_) => panic!("expected an explicit Sparse vector"),
        }
    }

    fn valid_request() -> ExactRrfQueryRequest {
        serde_json::from_value(serde_json::json!({
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
        .unwrap()
    }

    #[test]
    fn exact_rrf_request_accepts_the_frozen_two_channel_shape() {
        let request = valid_request();

        validate_exact_rrf_request(&request).unwrap();
    }

    #[test]
    fn explicit_replica_consistency_fails_closed() {
        require_default_consistency(false).unwrap();
        assert!(require_default_consistency(true).is_err());
    }

    #[test]
    fn exact_rrf_request_defaults_weights_and_accepts_empty_sparse() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1, 0.2], "using": "dense" },
                "sparse": {
                    "query": { "indices": [], "values": [] },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        }))
        .unwrap();

        assert_eq!(request.exact_rrf.weights, [1.0, 1.0]);
        validate_exact_rrf_request(&request).unwrap();
    }

    #[test]
    fn exact_rrf_request_rejects_unknown_fields() {
        let request = serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1], "using": "dense" },
                "sparse": {
                    "query": { "indices": [], "values": [] },
                    "using": "sparse"
                },
                "k": 60,
                "unknown": true
            },
            "limit": 20
        });

        assert!(serde_json::from_value::<ExactRrfQueryRequest>(request).is_err());
    }

    #[test]
    fn exact_rrf_rejects_invalid_dense_values() {
        let mut request = valid_request();
        request.exact_rrf.dense.query.clear();
        assert!(validate_exact_rrf_request(&request).is_err());

        request.exact_rrf.dense.query = vec![f32::INFINITY];
        assert!(validate_exact_rrf_request(&request).is_err());
    }

    #[test]
    fn exact_rrf_rejects_invalid_sparse_shape_and_order() {
        let mut request = valid_request();
        sparse_vector_mut(&mut request).values.pop();
        assert!(validate_exact_rrf_request(&request).is_err());

        request = valid_request();
        sparse_vector_mut(&mut request).indices = vec![12, 12];
        assert!(validate_exact_rrf_request(&request).is_err());

        sparse_vector_mut(&mut request).indices = vec![99, 12];
        assert!(validate_exact_rrf_request(&request).is_err());

        request = valid_request();
        sparse_vector_mut(&mut request).values[0] = f32::NAN;
        assert!(validate_exact_rrf_request(&request).is_err());
    }

    #[test]
    fn exact_rrf_accepts_and_resolves_local_bm25_documents() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1, 0.2], "using": "dense" },
                "sparse": {
                    "query": {
                        "text": "hybrid retrieval retrieval",
                        "model": "qdrant/bm25"
                    },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        }))
        .unwrap();

        validate_exact_rrf_request(&request).unwrap();
        let ExactRrfSparseQuery::Document(document) = request.exact_rrf.sparse.query else {
            panic!("expected a BM25 document");
        };
        let sparse = resolve_exact_sparse_query(ExactRrfSparseQuery::Document(document)).unwrap();
        assert!(!sparse.indices.is_empty());
        assert_eq!(sparse.indices.len(), sparse.values.len());
        assert!(sparse.indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            sparse
                .values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }

    #[test]
    fn exact_rrf_accepts_an_empty_bm25_document_as_an_empty_channel() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1], "using": "dense" },
                "sparse": {
                    "query": { "text": "", "model": "qdrant/bm25" },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        }))
        .unwrap();

        validate_exact_rrf_request(&request).unwrap();
        let sparse = resolve_exact_sparse_query(request.exact_rrf.sparse.query).unwrap();
        assert!(sparse.indices.is_empty());
        assert!(sparse.values.is_empty());
    }

    #[test]
    fn exact_rrf_bm25_document_rejects_non_local_models_and_unknown_options() {
        for query in [
            serde_json::json!({ "text": "query", "model": "remote/bm25" }),
            serde_json::json!({
                "text": "query",
                "model": "qdrant/bm25",
                "options": { "unrecognized": true }
            }),
        ] {
            let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
                "exact_rrf": {
                    "dense": { "query": [0.1], "using": "dense" },
                    "sparse": { "query": query, "using": "sparse" },
                    "k": 60
                },
                "limit": 20
            }))
            .unwrap();
            assert!(validate_exact_rrf_request(&request).is_err());
        }
    }

    #[test]
    fn exact_rrf_bm25_document_rejects_invalid_native_options() {
        let request: ExactRrfQueryRequest = serde_json::from_value(serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1], "using": "dense" },
                "sparse": {
                    "query": {
                        "text": "query",
                        "model": "qdrant/bm25",
                        "options": { "k": -1 }
                    },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        }))
        .unwrap();

        validate_exact_rrf_request(&request).unwrap();
        assert!(resolve_exact_sparse_query(request.exact_rrf.sparse.query).is_err());
    }

    #[test]
    fn exact_rrf_bm25_document_rejects_unknown_document_fields() {
        let request = serde_json::json!({
            "exact_rrf": {
                "dense": { "query": [0.1], "using": "dense" },
                "sparse": {
                    "query": {
                        "text": "query",
                        "model": "qdrant/bm25",
                        "remote": true
                    },
                    "using": "sparse"
                },
                "k": 60
            },
            "limit": 20
        });

        assert!(serde_json::from_value::<ExactRrfQueryRequest>(request).is_err());
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
    fn exact_rrf_rejects_invalid_weights_k_and_limit() {
        let mut request = valid_request();
        request.exact_rrf.weights = [0.0, 0.0];
        assert!(validate_exact_rrf_request(&request).is_err());

        request.exact_rrf.weights = [-1.0, 1.0];
        assert!(validate_exact_rrf_request(&request).is_err());

        request.exact_rrf.weights = [f32::NAN, 1.0];
        assert!(validate_exact_rrf_request(&request).is_err());

        request = valid_request();
        request.exact_rrf.k = 0;
        assert!(validate_exact_rrf_request(&request).is_err());

        request = valid_request();
        request.limit = 0;
        assert!(validate_exact_rrf_request(&request).is_err());
    }

    #[test]
    fn exact_rrf_response_serializes_the_frozen_v1_shape() {
        let response = ExactRrfQueryResponse {
            points: vec![ExactRrfHit {
                id: 42_u64.into(),
                rank: 1,
                version: 7,
            }],
            guarantee: ExactRrfGuarantee {
                scope: "selected-local-shards-frozen-segment-view",
                ordered_top_k_exact: true,
                tie_break: "point-identity-ascending",
                channel_input: "exact-channel-rank-streams",
            },
            execution: ExactRrfExecutionResponse {
                plan: "exact-rank-session-v1",
                stop_reason: "top-k-fixed",
                source_pulls: vec![31, 28],
                source_exhausted: vec![false, false],
                certification_checks: 9,
                corpus_points_observed: 1_000,
                query_rounds: 1,
                source_points_materialized: vec![81, 81],
                exhaustive_fallback: false,
            },
        };

        assert_eq!(
            serde_json::to_value(response).unwrap(),
            serde_json::json!({
                "points": [{ "id": 42, "rank": 1, "version": 7 }],
                "guarantee": {
                    "scope": "selected-local-shards-frozen-segment-view",
                    "orderedTopKExact": true,
                    "tieBreak": "point-identity-ascending",
                    "channelInput": "exact-channel-rank-streams"
                },
                "execution": {
                    "plan": "exact-rank-session-v1",
                    "stopReason": "top-k-fixed",
                    "sourcePulls": [31, 28],
                    "sourceExhausted": [false, false],
                    "certificationChecks": 9,
                    "corpusPointsObserved": 1000,
                    "queryRounds": 1,
                    "sourcePointsMaterialized": [81, 81],
                    "exhaustiveFallback": false
                }
            })
        );
    }

}
