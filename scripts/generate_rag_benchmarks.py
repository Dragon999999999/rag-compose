#!/usr/bin/env python3
"""Generate the checked-in RAG benchmark pipeline presets and manifest.

The benchmark suite is deliberately data-driven. This generator keeps the
experiment matrix, shared baselines, and reproducibility metadata in one
reviewable source while emitting ordinary composition-only YAML files that the
existing pipeline loader and CLI validate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# The experiment matrix contains long human-readable hypotheses and component
# declarations; keeping those values on one line makes the generator easier to
# compare with the manifest it emits.
# ruff: noqa: E501


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config/pipelines/benchmarks"
MANIFEST_PATH = ROOT / "benchmarks/rag_benchmark_manifest.yaml"

COMMON_QUERY = {
    "reasoner": "reasoner_basic_grounded",
    "renderer": "renderer_grounded_answer",
    "chat_history": {"mode": "recent_messages", "max_messages": 10},
}


def _query(
    name: str,
    retriever: str = "retriever_dense_vector_default",
    reranker: str = "reranker_identity",
    assessors: list[str] | None = None,
    decision: str = "decision_always_accept",
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one query-pipeline source document from registered IDs."""
    query: dict[str, Any] = {
        "retriever": retriever,
        "reranker": reranker,
        "reasoner": COMMON_QUERY["reasoner"],
        "assessors": assessors or ["assessor_noop"],
        "decision_policy": decision,
        "renderer": COMMON_QUERY["renderer"],
        "chat_history": COMMON_QUERY["chat_history"],
    }
    if execution is not None:
        query["execution"] = execution
    return {
        "schema_version": 2,
        "kind": "query",
        "name": name,
        "version": 1,
        "description": "Generated RAG benchmark query pipeline.",
        "query": query,
    }


def _ingestion(name: str, chunker: str, embedding: str, vector: str = "vector_backend_qdrant", knowledge: str = "knowledge_basic") -> dict[str, Any]:
    """Build one ingestion-pipeline source document from registered IDs."""
    return {
        "schema_version": 2,
        "kind": "ingestion",
        "name": name,
        "version": 1,
        "description": "Generated RAG benchmark ingestion pipeline.",
        "representation": {
            "parser": "parser_pymupdf",
            "knowledge_builder": knowledge,
            "chunker": chunker,
            "embedding": embedding,
            "vector_backend": vector,
        },
    }


def _alternative(
    alternative_id: str,
    *,
    branches: list[dict[str, Any]] | None = None,
    source: dict[str, Any] | None = None,
    stage_overrides: dict[str, Any] | None = None,
    decision: str = "basic_grounding",
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one unified-execution alternative."""
    value: dict[str, Any] = {
        "id": alternative_id,
        "acceptance": acceptance or {"strategy": "decision_policy"},
        "decision_policy": {
            "type": "decision_policy",
            "implementation": decision,
        },
    }
    if branches is not None:
        value["branches"] = branches
    if source is not None:
        value["source"] = source
    if stage_overrides is not None:
        value["stage_overrides"] = stage_overrides
    return value


def _execution(alternatives: list[dict[str, Any]], merge: str = "replace") -> dict[str, Any]:
    """Build bounded canonical alternative execution settings."""
    return {
        "max_attempts": len(alternatives),
        "max_stage_calls": 100,
        "evidence_merge": merge,
        "final_failure_policy": "last_result",
        "alternatives": alternatives,
    }


def _branch(branch_id: str, *, retriever: str | None = None, reranker: str | None = None, assessors: list[str] | None = None, start_at: str = "retriever") -> dict[str, Any]:
    """Build one branch using closed runtime implementation names."""
    value: dict[str, Any] = {"id": branch_id, "start_at": start_at}
    if retriever is not None:
        value["retriever"] = retriever
    if reranker is not None:
        value["reranker"] = reranker
    if assessors:
        value["assessors"] = assessors
    return value


def _record(
    experiment_id: str,
    name: str,
    family: str,
    ingestion: str | None,
    query: str | None,
    components: dict[str, Any],
    hypothesis: str,
    metrics: list[str],
    constraints: list[str],
    *,
    status: str = "supported",
    reason: str | None = None,
) -> dict[str, Any]:
    """Create one manifest entry with stable experiment metadata."""
    value: dict[str, Any] = {
        "id": experiment_id,
        "name": name,
        "family": family,
        "status": status,
        "ingestion_pipeline": ingestion,
        "query_pipeline": query,
        "components": components,
        "baseline": "BGE-M3 + recursive-token chunking + Qdrant + dense retrieval + identity reranker + noop assessor + always-accept",
        "hypothesis": hypothesis,
        "required_models_and_services": [
            "Registered local or configured remote model resources used by the selected components.",
            "PostgreSQL for persisted chunks and FTS where required; Qdrant, pgvector, or FAISS according to the ingestion preset.",
        ],
        "expected_metrics": metrics,
        "compatibility_constraints": constraints,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def build_suite(output_root: Path = DEFAULT_OUTPUT, manifest_path: Path = MANIFEST_PATH) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Generate all supported presets and all 76 manifest entries."""
    files: dict[str, Path] = {}
    entries: list[dict[str, Any]] = []

    family_names = {
        "01_baselines": "baselines",
        "02_chunking": "chunking",
        "03_embeddings": "embeddings",
        "04_vector_databases": "vector databases",
        "05_retrieval": "retrieval",
        "06_reranking": "reranking",
        "07_knowledge": "knowledge construction",
        "08_assessors": "assessors",
        "09_decision_policies": "decision policies",
        "10_interactions": "component interactions",
        "11_retrieval_branching": "retrieval branching",
        "12_reranker_branching": "reranker branching",
        "13_assessor_branching": "assessor branching",
        "14_recursive_alternatives": "recursive alternatives",
        "15_combined": "combined branching and recursion",
    }

    def write_pipeline(family_dir: str, filename: str, document: dict[str, Any]) -> str:
        path = output_root / family_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        files[document["name"]] = path
        return relative

    ingestion_refs: dict[str, str] = {}

    def add_ingestion(name: str, family_dir: str, chunker: str = "chunker_recursive_tokens", embedding: str = "embedding_local_bge_m3", vector: str = "vector_backend_qdrant", knowledge: str = "knowledge_basic") -> str:
        if name not in ingestion_refs:
            ingestion_refs[name] = write_pipeline(
                family_dir,
                f"{name}.yaml",
                _ingestion(name, chunker, embedding, vector, knowledge),
            )
        return ingestion_refs[name]

    def add_query(
        experiment_id: str,
        family_dir: str,
        ingestion: str | None,
        *,
        retriever: str = "retriever_dense_vector_default",
        reranker: str = "reranker_identity",
        assessors: list[str] | None = None,
        decision: str = "decision_always_accept",
        execution: dict[str, Any] | None = None,
        hypothesis: str = "Changing this stage should alter the selected quality or cost metric while other stages remain fixed.",
        components: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        constraints: list[str] | None = None,
        status: str = "supported",
        reason: str | None = None,
    ) -> None:
        name = f"benchmark_{experiment_id.lower().replace('-', '_')}"
        query_ref = write_pipeline(
            family_dir,
            f"{name}.yaml",
            _query(name, retriever, reranker, assessors, decision, execution),
        ) if status == "supported" else None
        entries.append(
            _record(
                experiment_id,
                name,
                family_names[family_dir],
                ingestion,
                query_ref,
                components or {},
                hypothesis,
                metrics or ["recall@K", "MRR", "answer correctness", "faithfulness", "end-to-end latency"],
                constraints or ["Use the same dataset, seed, prompts, retrieval depth, context budget, and generation settings across the family."],
                status=status,
                reason=reason,
            )
        )

    bge_ingestion = add_ingestion("benchmark_ingest_bge_recursive", "01_baselines")
    minilm_ingestion = add_ingestion("benchmark_ingest_minilm_recursive", "01_baselines", embedding="embedding_local_minilm")
    add_query("BASE-01", "01_baselines", minilm_ingestion, retriever="retriever_dense_vector_default", components={"embedding": "embedding_local_minilm"}, hypothesis="MiniLM provides a lightweight quality/latency reference.")
    add_query("BASE-02", "01_baselines", bge_ingestion, components={"embedding": "embedding_local_bge_m3"}, hypothesis="BGE-M3 improves retrieval quality over MiniLM at additional resource cost.")
    add_query("BASE-03", "01_baselines", bge_ingestion, retriever="retriever_hybrid_rrf_default", reranker="reranker_bge_v2_m3", assessors=["assessor_evidence_sufficiency"], decision="decision_basic_grounding", components={"retriever": "retriever_hybrid_rrf_default", "reranker": "reranker_bge_v2_m3", "assessor": "assessor_evidence_sufficiency"}, hypothesis="A stronger hybrid and reranking stack improves grounded answer quality.")

    chunkers = [
        ("CHUNK-01", "chunker_fixed_window"),
        ("CHUNK-02", "chunker_recursive_tokens"),
        ("CHUNK-03", "chunker_sentence_window"),
        ("CHUNK-04", "chunker_structural"),
        ("CHUNK-05", "chunker_block_aware"),
        ("CHUNK-06", "chunker_semantic"),
        ("CHUNK-07", "chunker_hierarchical"),
        ("CHUNK-08", "chunker_late"),
        ("CHUNK-09", "chunker_llm_guided"),
    ]
    for experiment_id, chunker in chunkers:
        if chunker == "chunker_late":
            add_query(experiment_id, "02_chunking", None, components={"chunker": chunker}, status="unsupported", reason="The registered late chunker requires embedding_jina_v2_base_en, but the repository has no composition component for that embedding model. Adding an unregistered alias would violate the benchmark constraint.")
            continue
        ingestion = add_ingestion(f"benchmark_ingest_{experiment_id.lower().replace('-', '_')}", "02_chunking", chunker=chunker)
        add_query(experiment_id, "02_chunking", ingestion, components={"chunker": chunker}, hypothesis=f"The {chunker} strategy changes boundary quality and retrieval context efficiency.", constraints=["Embedding, vector backend, query pipeline, and generation settings remain fixed.", "Specialized chunkers retain their registered model and parameter requirements."])

    for experiment_id, embedding in [("EMB-01", "embedding_local_minilm"), ("EMB-02", "embedding_local_bge_m3"), ("EMB-03", "embedding_local_gemma"), ("EMB-04", "embedding_local_nomic_v2"), ("EMB-05", "embedding_aqueduct_qwen")]:
        ingestion = add_ingestion(f"benchmark_ingest_{experiment_id.lower().replace('-', '_')}", "03_embeddings", embedding=embedding)
        add_query(experiment_id, "03_embeddings", ingestion, components={"embedding": embedding}, hypothesis="Changing the embedding model changes semantic recall and resource usage.", metrics=["recall@K", "nDCG@K", "context relevance", "embedding inference time", "estimated cost"])

    for experiment_id, vector in [("VDB-01", "vector_backend_qdrant"), ("VDB-02", "vector_backend_pgvector"), ("VDB-03", "vector_backend_faiss")]:
        ingestion = add_ingestion(f"benchmark_ingest_{experiment_id.lower().replace('-', '_')}", "04_vector_databases", vector=vector)
        add_query(experiment_id, "04_vector_databases", ingestion, components={"vector_backend": vector}, hypothesis=f"The {vector} backend changes retrieval latency, indexing throughput, and resource consumption.", metrics=["recall@K", "retrieval latency", "indexing throughput", "storage", "memory consumption"])

    retrievers = [
        ("RET-01", "retriever_dense_vector_default", "benchmark_ingest_bge_recursive", "dense vector"),
        ("RET-02", "retriever_dense_vector_summary", "benchmark_ingest_bge_recursive", "dense summary"),
        ("RET-03", "retriever_sparse_bm25", "benchmark_ingest_bge_recursive", "BM25"),
        ("RET-04", "retriever_sparse_postgres_fts", "benchmark_ingest_bge_recursive", "PostgreSQL FTS"),
        ("RET-05", "retriever_sparse_bge_m3", "benchmark_ingest_bge_recursive", "BGE-M3 sparse"),
        ("RET-06", "retriever_sparse_splade_v3", "benchmark_ingest_bge_recursive", "SPLADE-v3"),
        ("RET-07", "retriever_hybrid_rrf_default", "benchmark_ingest_bge_recursive", "hybrid RRF"),
        ("RET-08", "retriever_hybrid_weighted", "benchmark_ingest_bge_recursive", "weighted hybrid"),
        ("RET-09", "retriever_multivector_colbert", "benchmark_ingest_bge_recursive", "ColBERT multivector"),
        ("RET-10", "retriever_full_context_postgres", "benchmark_ingest_bge_recursive", "full context"),
    ]
    for experiment_id, retriever, _ingestion_name, label in retrievers:
        constraints = ["Use the same BGE-M3 recursive ingestion representation where compatible.", f"{label} retrieval requires the resources declared by its registered retriever contract."]
        add_query(experiment_id, "05_retrieval", bge_ingestion, retriever=retriever, components={"retriever": retriever}, hypothesis=f"The {label} strategy changes evidence recall, ranking quality, and query cost.", metrics=["recall@K", "precision@K", "MRR", "nDCG@K", "retrieval latency"], constraints=constraints)

    for experiment_id, reranker in [("RANK-01", "reranker_identity"), ("RANK-02", "reranker_top_k"), ("RANK-03", "reranker_cross_encoder"), ("RANK-04", "reranker_bge_v2_m3"), ("RANK-05", "reranker_jina_v2_multilingual"), ("RANK-06", "reranker_mxbai_base_v2"), ("RANK-07", "reranker_qwen3_0_6b")]:
        add_query(experiment_id, "06_reranking", bge_ingestion, reranker=reranker, components={"reranker": reranker}, hypothesis=f"The {reranker} changes ranking relevance and inference cost while preserving the baseline candidate pool.", metrics=["MRR", "nDCG@K", "context relevance", "reranker inference time", "memory consumption"])

    for experiment_id, knowledge in [("KNOW-01", "knowledge_basic"), ("KNOW-02", "knowledge_structural"), ("KNOW-03", "knowledge_metadata_rules"), ("KNOW-04", "knowledge_metadata_spacy"), ("KNOW-05", "knowledge_metadata_gliner")]:
        ingestion = add_ingestion(f"benchmark_ingest_{experiment_id.lower().replace('-', '_')}", "07_knowledge", knowledge=knowledge)
        add_query(experiment_id, "07_knowledge", ingestion, components={"knowledge_builder": knowledge}, hypothesis=f"The {knowledge} builder changes provenance and metadata available to retrieval and grounding.")

    for experiment_id, assessor in [("ASSESS-01", "assessor_noop"), ("ASSESS-02", "assessor_relevance"), ("ASSESS-03", "assessor_diversity"), ("ASSESS-04", "assessor_evidence_sufficiency"), ("ASSESS-05", "assessor_retrieval_confidence"), ("ASSESS-06", "assessor_source_reference")]:
        add_query(experiment_id, "08_assessors", bge_ingestion, assessors=[assessor], components={"assessor": assessor}, hypothesis=f"The {assessor} changes evidence validation and abstention behavior after generation.", metrics=["answer correctness", "faithfulness", "citation accuracy", "evidence sufficiency", "abstention accuracy"])

    decision_specs = [
        ("DEC-01", "decision_always_accept", ["assessor_noop"]),
        ("DEC-02", "decision_basic_grounding", ["assessor_evidence_sufficiency"]),
        ("DEC-03", "decision_threshold", ["assessor_relevance"]),
        ("DEC-04", "decision_consensus", ["assessor_relevance", "assessor_evidence_sufficiency"]),
        ("DEC-05", "decision_top_k", ["assessor_relevance"]),
        ("DEC-06", "decision_token_budget", ["assessor_evidence_sufficiency"]),
        ("DEC-07", "decision_select_branches", ["assessor_relevance"]),
        ("DEC-08", "decision_merge_branches", ["assessor_relevance"]),
        ("DEC-09", "decision_adaptive_retrieval", ["assessor_evidence_sufficiency"]),
        ("DEC-10", "decision_fallback_retrieval", ["assessor_evidence_sufficiency"]),
    ]
    branch_execution = _execution([_alternative("primary", branches=[_branch("dense", retriever="vector_similarity", assessors=["relevance"]), _branch("lexical", retriever="bm25", assessors=["relevance"])], decision="merge_branches", acceptance={"strategy": "always_accept"})], merge="union")
    for experiment_id, decision, assessors in decision_specs:
        execution = branch_execution if experiment_id in {"DEC-07", "DEC-08"} else None
        add_query(experiment_id, "09_decision_policies", bge_ingestion, assessors=assessors, decision=decision, execution=execution, components={"decision_policy": decision}, hypothesis=f"The {decision} changes acceptance, abstention, selection, or recovery behavior in its intended context.", metrics=["evidence sufficiency", "abstention accuracy", "answer correctness", "retrieval latency"], constraints=["Decision policies are evaluated with assessor outputs appropriate to their semantics.", "Recovery policies are bounded by max_attempts and max_stage_calls."])

    interaction_specs = [
        ("INT-01", "chunker_recursive_tokens", "retriever_dense_vector_default", "reranker_identity", bge_ingestion),
        ("INT-02", "chunker_semantic", "retriever_dense_vector_default", "reranker_identity", add_ingestion("benchmark_ingest_int_02_semantic", "10_interactions", chunker="chunker_semantic")),
        ("INT-03", "chunker_semantic", "retriever_dense_vector_default", "reranker_bge_v2_m3", add_ingestion("benchmark_ingest_int_03_semantic", "10_interactions", chunker="chunker_semantic")),
        ("INT-04", "chunker_hierarchical", "retriever_dense_vector_default", "reranker_bge_v2_m3", add_ingestion("benchmark_ingest_int_04_hierarchical", "10_interactions", chunker="chunker_hierarchical")),
        ("INT-05", "chunker_recursive_tokens", "retriever_hybrid_rrf_default", "reranker_bge_v2_m3", bge_ingestion),
        ("INT-06", "chunker_recursive_tokens", "retriever_sparse_bge_m3", "reranker_bge_v2_m3", bge_ingestion),
    ]
    for experiment_id, chunker, retriever, reranker, ingestion in interaction_specs:
        add_query(experiment_id, "10_interactions", ingestion, retriever=retriever, reranker=reranker, components={"chunker": chunker, "retriever": retriever, "reranker": reranker}, hypothesis="High-value stage combinations can reveal interaction effects that individual-stage comparisons miss.", constraints=["Keep all unlisted stages at their BGE-M3 baseline values.", "Learned sparse retrieval requires a separately prepared sparse index where the ingestion preset does not produce one automatically."])

    add_query("BRANCH-01", "11_retrieval_branching", bge_ingestion, assessors=["assessor_relevance"], decision="decision_merge_branches", execution=_execution([_alternative("primary", branches=[_branch("dense", retriever="vector_similarity", reranker="identity", assessors=["relevance"]), _branch("bm25", retriever="bm25", reranker="identity", assessors=["relevance"])], decision="merge_branches", acceptance={"strategy": "always_accept"})], merge="union"), components={"branches": ["vector_similarity", "bm25"], "decision_policy": "decision_merge_branches"}, hypothesis="Combining dense and lexical evidence improves recall and coverage.")
    add_query("BRANCH-02", "11_retrieval_branching", bge_ingestion, reranker="reranker_bge_v2_m3", assessors=["assessor_relevance"], decision="decision_select_branches", execution=_execution([_alternative("primary", branches=[_branch("dense", retriever="vector_similarity", reranker="cross_encoder", assessors=["relevance"]), _branch("hybrid", retriever="hybrid_rrf", reranker="cross_encoder", assessors=["relevance"])], decision="select_branches", acceptance={"strategy": "always_accept"})], merge="replace"), components={"branches": ["vector_similarity", "hybrid_rrf"], "reranker": "reranker_bge_v2_m3"}, hypothesis="A selective join can choose the stronger evidence path with lower context noise.", constraints=["Both branches use the configured BGE reranker implementation; branch-specific model parameters are inherited by the current execution architecture."])
    add_query("BRANCH-03", "11_retrieval_branching", bge_ingestion, assessors=["assessor_evidence_sufficiency"], decision="decision_merge_branches", execution=_execution([_alternative("primary", branches=[_branch("dense", retriever="vector_similarity", assessors=["evidence_sufficiency"]), _branch("bm25", retriever="bm25", assessors=["evidence_sufficiency"]), _branch("hybrid", retriever="hybrid_rrf", assessors=["evidence_sufficiency"])], decision="merge_branches", acceptance={"strategy": "always_accept"})], merge="union"), components={"branches": ["vector_similarity", "bm25", "hybrid_rrf"]}, hypothesis="Three complementary retrieval paths improve evidence coverage.")

    add_query("BRANCH-04", "12_reranker_branching", bge_ingestion, assessors=["assessor_relevance"], decision="decision_select_branches", execution=_execution([_alternative("primary", branches=[_branch("bge", reranker="cross_encoder", assessors=["relevance"]), _branch("jina", reranker="cross_encoder", assessors=["relevance"])], decision="select_branches", acceptance={"strategy": "always_accept"})]), components={"branches": ["reranker_bge_v2_m3", "reranker_jina_v2_multilingual"]}, hypothesis="Parallel reranking paths expose quality/latency trade-offs before branch selection.", constraints=["The current branch schema carries runtime implementation names, so model-specific branch settings are represented in the manifest and require branch-aware runtime configuration for exact BGE/Jina separation."])
    add_query("BRANCH-05", "12_reranker_branching", bge_ingestion, assessors=["assessor_relevance"], decision="decision_select_branches", execution=_execution([_alternative("primary", branches=[_branch("cross_encoder", reranker="cross_encoder", assessors=["relevance"]), _branch("qwen", reranker="generative", assessors=["relevance"])], decision="select_branches", acceptance={"strategy": "always_accept"})]), components={"branches": ["reranker_cross_encoder", "reranker_qwen3_0_6b"]}, hypothesis="A discriminative and generative reranker branch can trade latency for ranking quality.", constraints=["The current branch schema carries runtime implementation names; exact per-model branch settings require branch-aware runtime configuration."])

    add_query("BRANCH-06", "13_assessor_branching", bge_ingestion, assessors=["assessor_relevance"], decision="decision_basic_grounding", execution=_execution([_alternative("primary", branches=[_branch("relevance", start_at="assessor", assessors=["relevance"]), _branch("sufficiency", start_at="assessor", assessors=["evidence_sufficiency"])], decision="basic_grounding", acceptance={"strategy": "always_accept"})]), components={"branches": ["assessor_relevance", "assessor_evidence_sufficiency"]}, hypothesis="Independent assessor signals provide a more robust decision input than one signal alone.")
    add_query("BRANCH-07", "13_assessor_branching", bge_ingestion, assessors=["assessor_relevance"], decision="decision_basic_grounding", execution=_execution([_alternative("primary", branches=[_branch("diversity", start_at="assessor", assessors=["diversity"]), _branch("confidence", start_at="assessor", assessors=["retrieval_confidence"]), _branch("sufficiency", start_at="assessor", assessors=["evidence_sufficiency"])], decision="basic_grounding", acceptance={"strategy": "always_accept"})]), components={"branches": ["assessor_diversity", "assessor_retrieval_confidence", "assessor_evidence_sufficiency"]}, hypothesis="Diversity, confidence, and sufficiency together characterize retrieval quality more completely.")

    recursive_specs = [
        ("RECUR-01", [_alternative("primary", acceptance={"strategy": "evidence_sufficiency"}), _alternative("retry", source={"alternative_id": "primary", "path_id": "primary", "restart_stage": "assessor"}, stage_overrides={"assessors": [{"type": "assessor", "implementation": "retrieval_confidence"}]}, acceptance={"strategy": "always_accept"})], {"assessor": "assessor_evidence_sufficiency"}),
        ("RECUR-02", [_alternative("primary", stage_overrides={"assessors": [{"type": "assessor", "implementation": "relevance"}]}, acceptance={"strategy": "evidence_sufficiency"}), _alternative("retry", source={"alternative_id": "primary", "path_id": "primary", "restart_stage": "reranker"}, stage_overrides={"reranker": {"type": "reranker", "implementation": "cross_encoder", "model_id": "reranker_bge_v2_m3", "retrieval_top_k": 20, "rerank_top_k": 5, "device": "cuda", "batch_size": 2}, "assessors": [{"type": "assessor", "implementation": "evidence_sufficiency"}]}, acceptance={"strategy": "always_accept"})], {"reranker": "reranker_bge_v2_m3"}),
        ("RECUR-03", [_alternative("primary", stage_overrides={"assessors": [{"type": "assessor", "implementation": "relevance"}]}, acceptance={"strategy": "evidence_sufficiency"}), _alternative("retry", source={"alternative_id": "primary", "path_id": "primary", "restart_stage": "retriever"}, stage_overrides={"retriever": {"type": "hybrid", "implementation": "hybrid_rrf", "top_k": 10, "dense_top_k": 20, "sparse_top_k": 20, "rrf_k": 60, "query_embedder": None}, "assessors": [{"type": "assessor", "implementation": "evidence_sufficiency"}]}, acceptance={"strategy": "always_accept"})], {"retriever": "retriever_hybrid_rrf_default"}),
    ]
    for experiment_id, alternatives, changed in recursive_specs:
        add_query(experiment_id, "14_recursive_alternatives", bge_ingestion, assessors=["assessor_evidence_sufficiency"], decision="decision_basic_grounding", execution=_execution(alternatives), components=changed, hypothesis="A bounded retry should recover from insufficient evidence while reusing earlier outputs at the declared restart stage.", constraints=["Alternatives execute in order and stop on acceptance.", "Every retry has an explicit source path, decision policy, acceptance condition, max_attempts, and max_stage_calls."])

    adv_one = _execution([
        _alternative("primary", branches=[_branch("dense", retriever="vector_similarity", reranker="cross_encoder", assessors=["relevance"]), _branch("bm25", retriever="bm25", reranker="cross_encoder", assessors=["relevance"])], decision="select_branches", acceptance={"strategy": "evidence_sufficiency"}),
        _alternative("retry", source={"alternative_id": "primary", "path_id": "dense", "restart_stage": "assessor"}, stage_overrides={"reranker": {"type": "reranker", "implementation": "cross_encoder", "model_id": "reranker_bge_v2_m3", "retrieval_top_k": 20, "rerank_top_k": 5, "device": "cuda", "batch_size": 2}, "assessors": [{"type": "assessor", "implementation": "evidence_sufficiency"}]}, acceptance={"strategy": "always_accept"}),
    ])
    add_query("ADV-01", "15_combined", bge_ingestion, assessors=["assessor_evidence_sufficiency"], decision="decision_select_branches", execution=adv_one, components={"branches": ["vector_similarity", "bm25"], "retry_source": "primary/dense", "retry_reranker": "reranker_bge_v2_m3"}, hypothesis="A selected branch can be partially retried with stronger reranking without repeating retrieval.", constraints=["The retry explicitly reuses primary/dense output and restarts at assessor.", "Branch-specific model settings follow the current branch-aware runtime limitation."])

    adv_two = _execution([
        _alternative("dense", acceptance={"strategy": "evidence_sufficiency"}),
        _alternative("hybrid", source={"alternative_id": "dense", "path_id": "primary", "restart_stage": "retriever"}, stage_overrides={"retriever": {"type": "hybrid", "implementation": "hybrid_rrf", "top_k": 10, "dense_top_k": 20, "sparse_top_k": 20, "rrf_k": 60, "query_embedder": None}}, acceptance={"strategy": "evidence_sufficiency"}),
        _alternative("reranker_branches", source={"alternative_id": "hybrid", "path_id": "primary", "restart_stage": "reranker"}, branches=[_branch("bge", reranker="cross_encoder", assessors=["relevance"]), _branch("jina", reranker="cross_encoder", assessors=["evidence_sufficiency"])], decision="merge_branches", acceptance={"strategy": "always_accept"}),
    ])
    add_query("ADV-02", "15_combined", bge_ingestion, assessors=["assessor_evidence_sufficiency"], decision="decision_merge_branches", execution=adv_two, components={"alternatives": ["dense", "hybrid", "reranker_branches"], "retriever": "retriever_hybrid_rrf_default", "rerankers": ["reranker_bge_v2_m3", "reranker_jina_v2_multilingual"]}, hypothesis="Sequential retrieval alternatives followed by a final reranker branch maximize recovery coverage under a bounded budget.", constraints=["Alternatives stop after acceptance and reuse the explicitly named earlier path.", "Exact model-specific branch settings require branch-aware runtime configuration."])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump({"schema_version": 1, "suite": "rag-benchmark-76", "seed": 0, "evaluation_conditions": {"dataset": "Use the same fixed evaluation corpus and query set for every family.", "prompts": "Use the selected registered prompt unchanged across comparisons.", "stochasticity": "Record model/provider stochasticity separately from deterministic pipeline metrics."}, "experiments": entries}, sort_keys=False), encoding="utf-8")
    return entries, files


def main() -> int:
    """Generate benchmark YAML and manifest files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    entries, files = build_suite(args.output_root, args.manifest)
    supported = sum(entry["status"] == "supported" for entry in entries)
    print(f"generated {len(entries)} experiment entries ({supported} with YAML pipelines, {len(files)} unique pipeline files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
