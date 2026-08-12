"""Budgeted, observer-assisted selection for tool and skill context assets.

This module deliberately stops at producing a selection plan.  The canonical catalog
stays local: an observer sees only bounded cards and may rank known IDs, but it cannot
rewrite a tool schema, a skill, mandatory membership, or dependency edges.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence


_KINDS = {"tool", "skill"}
_SELECTOR_OUTPUT_KEYS = {"catalog_digest", "ranked_asset_ids", "confidence"}
_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("context asset data must be canonical-JSON serializable") from exc


@dataclass(frozen=True)
class ContextAsset:
    """One canonical prompt asset with a cheaper structural representation.

    ``structural_cost`` uses whichever unit the caller's budget uses (normally exact
    tokenizer tokens or a conservative character estimate).  It is intentionally
    explicit so this core does not pretend that JSON character count equals tokens.
    """

    asset_id: str
    kind: str
    canonical_payload: Any
    structural_payload: Any
    structural_cost: int
    dependencies: tuple[str, ...] = ()
    mandatory: bool = False
    card: str = ""
    recency: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        if self.asset_id != self.asset_id.strip():
            raise ValueError("asset_id must not contain surrounding whitespace")
        if self.kind not in _KINDS:
            raise ValueError("asset kind must be 'tool' or 'skill'")
        if isinstance(self.structural_cost, bool) or not isinstance(self.structural_cost, int):
            raise ValueError("structural_cost must be an integer")
        if self.structural_cost < 0:
            raise ValueError("structural_cost must be non-negative")
        if not isinstance(self.card, str):
            raise ValueError("card must be a string")
        if isinstance(self.recency, bool) or not isinstance(self.recency, int):
            raise ValueError("recency must be an integer")

        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("dependencies must contain non-empty asset IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"duplicate dependency on asset {self.asset_id!r}")
        if self.asset_id in dependencies:
            raise ValueError(f"dependency cycle includes asset {self.asset_id!r}")
        object.__setattr__(self, "dependencies", dependencies)


@dataclass(frozen=True)
class SelectorAsset:
    """Bounded observer view; canonical payloads are intentionally absent."""

    asset_id: str
    kind: str
    card: str
    structural_cost: int
    dependencies: tuple[str, ...]
    recency: int


@dataclass(frozen=True)
class SelectorRequest:
    """The only input supplied to an optional semantic selector callback."""

    assets: tuple[SelectorAsset, ...]
    mandatory_ids: tuple[str, ...]
    query: str
    budget: int
    remaining_budget: int
    catalog_digest: str
    action_state_digest: str
    # Canonical JSON of the caller-provided bounded state view.  This is what makes C's
    # ranking sensitive to the latest observation/error/live artifact without exposing
    # canonical tool schemas or skill bodies.
    action_state: str


@dataclass(frozen=True)
class SelectionPlan:
    """Locally validated projection plan returned to the future renderer."""

    catalog_digest: str
    action_state_digest: str
    cache_key: str
    selected_ids: tuple[str, ...]
    mandatory_ids: tuple[str, ...]
    resolutions: dict[str, str]
    total_cost: int
    budget: int
    over_budget: bool
    dropped_ids: tuple[str, ...]
    reason: str
    cache_status: str
    selector_confidence: float | None
    fallback_metadata: dict[str, Any]


SelectorCallback = Callable[[SelectorRequest], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class _CacheEntry:
    plan: SelectionPlan
    expires_at: float


def _catalog_by_id(assets: Sequence[ContextAsset]) -> dict[str, ContextAsset]:
    by_id: dict[str, ContextAsset] = {}
    for asset in assets:
        if not isinstance(asset, ContextAsset):
            raise ValueError("catalog entries must be ContextAsset instances")
        if asset.asset_id in by_id:
            raise ValueError(f"duplicate asset_id: {asset.asset_id}")
        by_id[asset.asset_id] = asset

    for asset in by_id.values():
        for dependency in asset.dependencies:
            if dependency not in by_id:
                raise ValueError(
                    f"unknown dependency {dependency!r} for asset {asset.asset_id!r}"
                )

    # Reject cycles up front.  A dormant optional cycle must not become a production-only
    # failure merely because the observer happened to select it on one request.
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(asset_id: str) -> None:
        if asset_id in visiting:
            raise ValueError(f"dependency cycle includes asset {asset_id!r}")
        if asset_id in visited:
            return
        visiting.add(asset_id)
        for dependency in sorted(by_id[asset_id].dependencies):
            visit(dependency)
        visiting.remove(asset_id)
        visited.add(asset_id)

    for asset_id in sorted(by_id):
        visit(asset_id)
    return by_id


def canonical_catalog_digest(assets: Sequence[ContextAsset]) -> str:
    """Hash the complete, order-independent catalog and all selection inputs."""

    by_id = _catalog_by_id(tuple(assets))
    canonical = [
        {
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "canonical_payload": asset.canonical_payload,
            "structural_payload": asset.structural_payload,
            "structural_cost": asset.structural_cost,
            "dependencies": sorted(asset.dependencies),
            "mandatory": asset.mandatory,
            "card": asset.card,
            "recency": asset.recency,
        }
        for asset in (by_id[asset_id] for asset_id in sorted(by_id))
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _dependency_order(
    by_id: Mapping[str, ContextAsset], seeds: Iterable[str]
) -> tuple[str, ...]:
    order: list[str] = []
    visited: set[str] = set()

    def visit(asset_id: str) -> None:
        if asset_id not in by_id:
            raise ValueError(f"unknown mandatory asset_id: {asset_id}")
        if asset_id in visited:
            return
        for dependency in sorted(by_id[asset_id].dependencies):
            visit(dependency)
        visited.add(asset_id)
        order.append(asset_id)

    for seed in sorted(set(seeds)):
        visit(seed)
    return tuple(order)


def _action_state_digest(
    action_state: Any, query: str, mandatory_ids: Iterable[str]
) -> str:
    value = {
        "action_state": action_state,
        "query": query,
        "mandatory_ids": sorted(set(mandatory_ids)),
    }
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _selection_cache_key(
    *,
    catalog_digest: str,
    action_state_digest: str,
    budget: int,
    budget_bucket: str,
    selector_contract: str,
    selector_model: str,
    codec: str,
    tenant: str,
) -> str:
    value = {
        "catalog_digest": catalog_digest,
        "action_state_digest": action_state_digest,
        "budget": budget,
        "budget_bucket": budget_bucket,
        "selector_contract": selector_contract,
        "selector_model": selector_model,
        "codec": codec,
        "tenant": tenant,
    }
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fallback_ranking(
    assets: Sequence[ContextAsset], query: str
) -> tuple[str, ...]:
    query_terms = set(_WORD_RE.findall(query.casefold()))

    def rank(asset: ContextAsset) -> tuple[int, int, str]:
        text = f"{asset.asset_id} {asset.kind} {asset.card}".casefold()
        overlap = len(query_terms.intersection(_WORD_RE.findall(text)))
        return (-overlap, -asset.recency, asset.asset_id)

    return tuple(asset.asset_id for asset in sorted(assets, key=rank))


def _validate_selector_output(
    raw: Any, known_ids: set[str], catalog_digest: str
) -> tuple[tuple[str, ...], float]:
    if not isinstance(raw, Mapping) or set(raw) != _SELECTOR_OUTPUT_KEYS:
        raise ValueError(
            "selector output must contain only catalog_digest, ranked_asset_ids, and confidence"
        )
    if raw.get("catalog_digest") != catalog_digest:
        raise ValueError("selector output was produced for a stale catalog")
    ranked = raw.get("ranked_asset_ids")
    confidence = raw.get("confidence")
    if not isinstance(ranked, list):
        raise ValueError("ranked_asset_ids must be a JSON list")
    if any(not isinstance(asset_id, str) or not asset_id for asset_id in ranked):
        raise ValueError("ranked_asset_ids must contain non-empty strings")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_asset_ids must not contain duplicates")
    if any(asset_id not in known_ids for asset_id in ranked):
        raise ValueError("ranked_asset_ids contains an unknown asset")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return tuple(ranked), confidence_value


def _clone_with_cache_status(plan: SelectionPlan, status: str) -> SelectionPlan:
    # Avoid sharing mutable metadata dictionaries between cache consumers.
    return replace(
        plan,
        cache_status=status,
        resolutions=dict(plan.resolutions),
        fallback_metadata=dict(plan.fallback_metadata),
    )


class BudgetedAssetResolver:
    """Select context assets under a hard budget without delegating truth to C."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.6,
        success_ttl: float = 300.0,
        failure_ttl: float = 2.0,
        max_failure_ttl: float = 30.0,
        max_cache_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if success_ttl < 0 or failure_ttl < 0 or max_failure_ttl < failure_ttl:
            raise ValueError("invalid cache TTL configuration")
        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be positive")
        self._min_confidence = min_confidence
        self._success_ttl = success_ttl
        self._failure_ttl = failure_ttl
        self._max_failure_ttl = max_failure_ttl
        self._max_cache_entries = max_cache_entries
        self._clock = clock
        self._cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._failure_counts: "OrderedDict[str, int]" = OrderedDict()
        self._inflight: dict[str, "asyncio.Task[SelectionPlan]"] = {}

    def _cache_get(self, key: str) -> SelectionPlan | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return _clone_with_cache_status(entry.plan, "hit")

    def _cache_put(self, key: str, plan: SelectionPlan, ttl: float) -> None:
        # An empty plan must be reconsidered next turn: it may reflect a transient C
        # failure or a budget boundary, and caching it would create a silent dead end.
        if not plan.selected_ids or ttl <= 0:
            return
        stored_plan = _clone_with_cache_status(plan, plan.cache_status)
        self._cache[key] = _CacheEntry(plan=stored_plan, expires_at=self._clock() + ttl)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)

    def _record_failure(self, key: str) -> tuple[int, float]:
        attempt = self._failure_counts.get(key, 0) + 1
        self._failure_counts[key] = attempt
        self._failure_counts.move_to_end(key)
        while len(self._failure_counts) > self._max_cache_entries:
            self._failure_counts.popitem(last=False)
        # The cap prevents an unbounded big-int exponent after a long outage.
        exponent = min(attempt - 1, 30)
        ttl = min(self._max_failure_ttl, self._failure_ttl * (2 ** exponent))
        return attempt, ttl

    def _record_success(self, key: str) -> None:
        self._failure_counts.pop(key, None)

    @staticmethod
    def _pack(
        by_id: Mapping[str, ContextAsset],
        mandatory_order: tuple[str, ...],
        ranking: Iterable[str],
        budget: int,
    ) -> tuple[tuple[str, ...], int]:
        selected_order = list(mandatory_order)
        selected = set(selected_order)
        total_cost = sum(by_id[asset_id].structural_cost for asset_id in selected)

        for asset_id in ranking:
            closure_order = _dependency_order(by_id, (asset_id,))
            additions = [item for item in closure_order if item not in selected]
            addition_cost = sum(by_id[item].structural_cost for item in additions)
            if total_cost + addition_cost > budget:
                continue
            selected_order.extend(additions)
            selected.update(additions)
            total_cost += addition_cost
        return tuple(selected_order), total_cost

    @staticmethod
    def _plan(
        *,
        by_id: Mapping[str, ContextAsset],
        catalog_digest: str,
        action_digest: str,
        cache_key: str,
        selected_ids: tuple[str, ...],
        mandatory_ids: tuple[str, ...],
        budget: int,
        reason: str,
        cache_status: str = "bypass",
        selector_confidence: float | None = None,
        fallback_metadata: Mapping[str, Any] | None = None,
    ) -> SelectionPlan:
        selected_set = set(selected_ids)
        total_cost = sum(by_id[asset_id].structural_cost for asset_id in selected_set)
        return SelectionPlan(
            catalog_digest=catalog_digest,
            action_state_digest=action_digest,
            cache_key=cache_key,
            selected_ids=selected_ids,
            mandatory_ids=mandatory_ids,
            resolutions={asset_id: "structural" for asset_id in selected_ids},
            total_cost=total_cost,
            budget=budget,
            over_budget=total_cost > budget,
            dropped_ids=tuple(sorted(set(by_id).difference(selected_set))),
            reason=reason,
            cache_status=cache_status,
            selector_confidence=selector_confidence,
            fallback_metadata=dict(fallback_metadata or {"used": False}),
        )

    async def resolve(
        self,
        assets: Sequence[ContextAsset],
        *,
        budget: int,
        action_state: Any,
        query: str = "",
        mandatory_ids: Iterable[str] = (),
        selector: SelectorCallback | None = None,
        budget_bucket: str = "",
        selector_contract: str = "context-asset-selector.v1",
        selector_model: str = "unspecified",
        codec: str = "unspecified",
        tenant: str = "default",
    ) -> SelectionPlan:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer")
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        asset_tuple = tuple(assets)
        by_id = _catalog_by_id(asset_tuple)
        catalog_digest = canonical_catalog_digest(asset_tuple)
        mandatory_seeds = set(mandatory_ids)
        mandatory_seeds.update(
            asset.asset_id for asset in asset_tuple if asset.mandatory
        )
        mandatory_order = _dependency_order(by_id, mandatory_seeds)
        action_digest = _action_state_digest(action_state, query, mandatory_seeds)
        bucket = budget_bucket or f"exact:{budget}"
        cache_key = _selection_cache_key(
            catalog_digest=catalog_digest,
            action_state_digest=action_digest,
            budget=budget,
            budget_bucket=bucket,
            selector_contract=selector_contract,
            selector_model=selector_model,
            codec=codec,
            tenant=tenant,
        )

        all_ids = tuple(sorted(by_id))
        full_cost = sum(asset.structural_cost for asset in by_id.values())
        if full_cost <= budget:
            return self._plan(
                by_id=by_id,
                catalog_digest=catalog_digest,
                action_digest=action_digest,
                cache_key=cache_key,
                selected_ids=all_ids,
                mandatory_ids=mandatory_order,
                budget=budget,
                reason="catalog_fits_budget",
            )

        mandatory_cost = sum(by_id[item].structural_cost for item in mandatory_order)
        if mandatory_cost > budget:
            return self._plan(
                by_id=by_id,
                catalog_digest=catalog_digest,
                action_digest=action_digest,
                cache_key=cache_key,
                selected_ids=mandatory_order,
                mandatory_ids=mandatory_order,
                budget=budget,
                reason="mandatory_closure_exceeds_budget",
            )
        if mandatory_cost == budget:
            # No semantic trade-off remains, but zero-cost cards and their already-paid
            # dependencies still fit.  Include them deterministically without calling C.
            selected_ids, _ = self._pack(by_id, mandatory_order, sorted(by_id), budget)
            return self._plan(
                by_id=by_id,
                catalog_digest=catalog_digest,
                action_digest=action_digest,
                cache_key=cache_key,
                selected_ids=selected_ids,
                mandatory_ids=mandatory_order,
                budget=budget,
                reason="mandatory_closure_consumes_budget",
            )

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        existing = self._inflight.get(cache_key)
        if existing is not None:
            shared = await asyncio.shield(existing)
            return _clone_with_cache_status(shared, "singleflight")

        task = asyncio.create_task(self._resolve_uncached(
            asset_tuple=asset_tuple,
            by_id=by_id,
            mandatory_order=mandatory_order,
            budget=budget,
            action_state=action_state,
            query=query,
            selector=selector,
            catalog_digest=catalog_digest,
            action_digest=action_digest,
            cache_key=cache_key,
        ))
        self._inflight[cache_key] = task

        def clear_inflight(done: "asyncio.Task[SelectionPlan]") -> None:
            if self._inflight.get(cache_key) is done:
                self._inflight.pop(cache_key, None)

        task.add_done_callback(clear_inflight)
        return await asyncio.shield(task)

    async def _resolve_uncached(
        self,
        *,
        asset_tuple: tuple[ContextAsset, ...],
        by_id: Mapping[str, ContextAsset],
        mandatory_order: tuple[str, ...],
        budget: int,
        action_state: Any,
        query: str,
        selector: SelectorCallback | None,
        catalog_digest: str,
        action_digest: str,
        cache_key: str,
    ) -> SelectionPlan:
        mandatory_cost = sum(by_id[item].structural_cost for item in mandatory_order)
        request = SelectorRequest(
            assets=tuple(
                SelectorAsset(
                    asset_id=asset.asset_id,
                    kind=asset.kind,
                    card=asset.card,
                    structural_cost=asset.structural_cost,
                    dependencies=asset.dependencies,
                    recency=asset.recency,
                )
                for asset in sorted(asset_tuple, key=lambda item: item.asset_id)
            ),
            mandatory_ids=mandatory_order,
            query=query,
            budget=budget,
            remaining_budget=budget - mandatory_cost,
            catalog_digest=catalog_digest,
            action_state_digest=action_digest,
            action_state=_canonical_json(action_state),
        )

        fallback_reason = ""
        confidence: float | None = None
        ranking: tuple[str, ...]
        if selector is None:
            fallback_reason = "selector_unavailable"
        else:
            try:
                raw = await selector(request)
            except Exception:
                # Error class/message may contain provider URLs or credentials; the plan
                # exposes only this stable category.
                fallback_reason = "selector_exception"
            else:
                try:
                    ranking, confidence = _validate_selector_output(
                        raw, set(by_id), catalog_digest
                    )
                except ValueError:
                    fallback_reason = "invalid_selector_output"
            if not fallback_reason and confidence is not None:
                if confidence < self._min_confidence:
                    fallback_reason = "low_confidence"

        if fallback_reason:
            lexical_query = query.strip() or _canonical_json(action_state)
            ranking = _fallback_ranking(asset_tuple, lexical_query)
            selected_ids, _ = self._pack(
                by_id, mandatory_order, ranking, budget
            )
            attempt, ttl = self._record_failure(cache_key)
            plan = self._plan(
                by_id=by_id,
                catalog_digest=catalog_digest,
                action_digest=action_digest,
                cache_key=cache_key,
                selected_ids=selected_ids,
                mandatory_ids=mandatory_order,
                budget=budget,
                reason="deterministic_fallback",
                cache_status="miss",
                selector_confidence=confidence,
                fallback_metadata={
                    "used": True,
                    "reason": fallback_reason,
                    "strategy": "lexical_then_recent",
                    "attempt_count": attempt,
                    "retry_after_seconds": ttl,
                },
            )
            self._cache_put(cache_key, plan, ttl)
            return plan

        self._record_success(cache_key)
        selected_ids, _ = self._pack(by_id, mandatory_order, ranking, budget)
        plan = self._plan(
            by_id=by_id,
            catalog_digest=catalog_digest,
            action_digest=action_digest,
            cache_key=cache_key,
            selected_ids=selected_ids,
            mandatory_ids=mandatory_order,
            budget=budget,
            reason="selector_ranked",
            cache_status="miss",
            selector_confidence=confidence,
        )
        self._cache_put(cache_key, plan, self._success_ttl)
        return plan
