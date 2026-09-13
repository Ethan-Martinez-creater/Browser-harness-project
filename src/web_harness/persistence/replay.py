"""Offline trace replay (Phase 2A1).

Replay reads an existing run directory and validates it entirely offline:

    run directory
       ↓
    TraceBundleLoader
       ↓
    TraceReplayEngine
       ↓
    ReplayReport

Hard guarantees:

- zero model calls (no ModelAdapter, no DecisionExecutor);
- zero environment actions (no EnvironmentAdapter.reset/step, no BrowserGym);
- zero mutations of the original trace (read-only);
- structural violations never raise: they are collected into a structured
  ReplayReport.

Two replay levels:

- Structural replay (all trace versions): layout and consistency checks over
  manifest.json, steps.jsonl, events.jsonl, result.json and the artifact
  references, plus metric recomputation and the recovery/replan outcome
  invariants.
- Semantic verification replay (schema v2 only): re-runs the deterministic
  StepVerifier over the recorded pre/post Observations and compares the
  regenerated FailureSignals with the recorded verification events. Only the
  StepVerifier is replayed — never the LLM, the policy control flow, the
  RecoveryManager or the Replanner.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from web_harness.core.events import RuntimeEvent, RuntimeEventType
from web_harness.core.models import EnvironmentStep, Observation, RunResult, StepRecord, TaskSpec
from web_harness.core.reliability import ReliabilityState
from web_harness.persistence.schema import parse_manifest_schema_version

# StepRecord reference fields that must point at existing files
_ARTIFACT_REF_FIELDS = (
    "observation_ref",
    "next_observation_ref",
    "observation_json_ref",
    "next_observation_json_ref",
    "prompt_ref",
    "model_response_ref",
)

REBUILDABLE_VERIFIER_IMPLEMENTATION = "DefaultStepVerifier"


class VerifierSpec(BaseModel):
    """Strict verification_spec contract (Phase 2A1 closure).

    Strict typing on purpose: no Python truthiness, no silent coercion, no
    default guessing. `"false"` is not a bool, `"2"` is not a threshold, a
    missing field is malformed — every deviation is a structured replay
    error instead of a fallback to defaults.
    """

    model_config = ConfigDict(strict=True)

    implementation: str
    detect_no_progress: bool
    detect_loop: bool
    loop_consecutive_threshold: int = Field(ge=1)


class TraceBundle(BaseModel):
    """Everything one run directory contains, loaded read-only.

    `load_errors` lists files/lines that could not be parsed; the bundle is
    still returned so replay can report the damage structurally instead of
    crashing.
    """

    run_dir: str
    trace_schema_version: int
    manifest: dict = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    events: list[RuntimeEvent] = Field(default_factory=list)
    result: RunResult | None = None
    load_errors: list[str] = Field(default_factory=list)

    @property
    def run_id(self) -> str | None:
        """Canonical run id: manifest first, then result, then first step."""
        if self.manifest.get("run_id"):
            return str(self.manifest["run_id"])
        if self.result is not None:
            return self.result.run_id
        if self.steps:
            return self.steps[0].run_id
        return None


class TraceBundleLoader:
    """Read-only loader for one run directory. Never writes or fixes files."""

    def load(self, run_dir: Path) -> TraceBundle:
        run_dir = Path(run_dir)
        errors: list[str] = []

        manifest: dict = {}
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            errors.append("manifest.json missing")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"manifest.json unparsable: {exc}")
                manifest = {}
            if not isinstance(manifest, dict):
                # parse succeeded but the shape is wrong: fail closed with a
                # structured error instead of tripping model validation
                errors.append("manifest.json top level is not a JSON object")
                manifest = {}

        schema_version, schema_errors = parse_manifest_schema_version(manifest)
        errors.extend(schema_errors)

        steps = self._read_jsonl(
            run_dir / "steps.jsonl",
            StepRecord,
            "steps.jsonl",
            errors,
            required=True,
        )
        events = self._read_jsonl(
            run_dir / "events.jsonl", RuntimeEvent, "events.jsonl", errors
        )

        result: RunResult | None = None
        result_path = run_dir / "result.json"
        if result_path.exists():
            try:
                result = RunResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
                errors.append(f"result.json unparsable: {exc}")

        return TraceBundle(
            run_dir=str(run_dir),
            trace_schema_version=schema_version,
            manifest=manifest,
            steps=steps,
            events=events,
            result=result,
            load_errors=errors,
        )

    @staticmethod
    def _read_jsonl(
        path: Path,
        model_cls: type[BaseModel],
        label: str,
        errors: list[str],
        *,
        required: bool = False,
    ) -> list:
        if not path.exists():
            if required:
                errors.append(f"{label} missing")
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{label} unreadable: {exc}")
            return []
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(model_cls.model_validate_json(line))
            except ValidationError as exc:
                errors.append(f"{label} line {line_no} unparsable: {exc}")
        return records

    # -- lazy artifact readers --------------------------------------------

    def read_observation(self, bundle: TraceBundle, ref: str) -> Observation:
        """Load one structured Observation artifact by its run-relative ref."""
        path = Path(bundle.run_dir) / ref
        return Observation.model_validate_json(path.read_text(encoding="utf-8"))


class ReplayReport(BaseModel):
    """Structured outcome of one offline replay (never raises on bad traces)."""

    run_id: str | None
    trace_schema_version: int
    structural_valid: bool = True

    step_count: int = 0
    event_count: int = 0

    artifact_missing_count: int = 0
    invariant_error_count: int = 0

    metric_mismatches: list[str] = Field(default_factory=list)
    verification_mismatches: list[str] = Field(default_factory=list)

    replayed_verifications: int = 0
    semantic_replay_skipped: bool = False
    # None = semantic replay did not run (legacy trace, --no-semantic, or no
    # rebuildable verifier spec); True = ran and matched; False = ran and
    # found mismatches or hit semantic-stage errors (fail-closed)
    semantic_valid: bool | None = None

    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def overall_valid(self) -> bool:
        """CLI-facing verdict: any real replay failure fails the run."""
        return self.structural_valid and self.semantic_valid is not False


class TraceReplayEngine:
    """Offline replay engine: structural validation + semantic verifier replay.

    The verifier is resolved in this order:

    1. an explicitly injected `verifier` (programmatic use / tests);
    2. the recorded `verification_spec` from the run manifest (schema v2) —
       the effective DefaultStepVerifier configuration the live run used;
    3. no verifier: semantic replay is skipped with an explicit note (a v2
       trace with verification events but no reusable spec is NEVER silently
       replayed against default settings).
    """

    def __init__(self, *, verifier=None, semantic_replay: bool = True):
        self._verifier = verifier
        self._semantic_replay = semantic_replay
        self._loader = TraceBundleLoader()

    def run(self, bundle: TraceBundle) -> ReplayReport:
        report = ReplayReport(
            run_id=bundle.run_id,
            trace_schema_version=bundle.trace_schema_version,
            step_count=len(bundle.steps),
            event_count=len(bundle.events),
        )
        self._structural_replay(bundle, report)
        report.structural_valid = not report.errors and not report.metric_mismatches
        if self._semantic_replay:
            errors_before = len(report.errors)
            self._semantic_verification_replay(bundle, report)
            if report.semantic_replay_skipped:
                report.semantic_valid = None
            else:
                # fail-closed: semantic-stage errors (e.g. corrupt structured
                # observations) or mismatches make the semantic pass invalid
                report.semantic_valid = (
                    len(report.errors) == errors_before
                    and not report.verification_mismatches
                )
        else:
            report.semantic_replay_skipped = True
            report.semantic_valid = None
        return report

    # -- structural replay ---------------------------------------------------

    def _structural_replay(self, bundle: TraceBundle, report: ReplayReport) -> None:
        report.errors.extend(bundle.load_errors)

        run_id = bundle.run_id
        # step_index contiguous from 0, in recorded order
        for position, step in enumerate(bundle.steps):
            if step.step_index != position:
                report.errors.append(
                    f"steps.jsonl step_index not contiguous: position {position} "
                    f"has step_index {step.step_index}"
                )
                break
        # run_id consistency across steps / events / result
        if run_id is not None:
            for step in bundle.steps:
                if step.run_id != run_id:
                    report.errors.append(
                        f"step {step.step_index} run_id mismatch: "
                        f"{step.run_id!r} != {run_id!r}"
                    )
            for event in bundle.events:
                if event.run_id != run_id:
                    report.errors.append(
                        f"event {event.event_id} run_id mismatch: "
                        f"{event.run_id!r} != {run_id!r}"
                    )
            if bundle.result is not None and bundle.result.run_id != run_id:
                report.errors.append(
                    f"result.json run_id mismatch: "
                    f"{bundle.result.run_id!r} != {run_id!r}"
                )
        # event step_index must reference a real recorded step
        for event in bundle.events:
            if event.step_index is not None and not (
                0 <= event.step_index < len(bundle.steps)
            ):
                report.errors.append(
                    f"event {event.event_id} references unknown step_index "
                    f"{event.step_index}"
                )

        # artifact references must exist on disk
        missing = 0
        for step in bundle.steps:
            for field in _ARTIFACT_REF_FIELDS:
                ref = getattr(step, field)
                if ref is not None and not (Path(bundle.run_dir) / ref).exists():
                    missing += 1
                    report.errors.append(
                        f"step {step.step_index} {field} missing: {ref}"
                    )
        report.artifact_missing_count = missing

        # result-level recomputable metrics and outcome invariants
        result = bundle.result
        if result is not None:
            step_input = sum(s.input_tokens or 0 for s in bundle.steps)
            step_output = sum(s.output_tokens or 0 for s in bundle.steps)
            expected_input = step_input + result.replan_input_tokens
            expected_output = step_output + result.replan_output_tokens
            if result.input_tokens != expected_input:
                report.metric_mismatches.append(
                    f"input_tokens: result={result.input_tokens} "
                    f"recomputed={expected_input}"
                )
            if result.output_tokens != expected_output:
                report.metric_mismatches.append(
                    f"output_tokens: result={result.output_tokens} "
                    f"recomputed={expected_output}"
                )
            if result.num_steps != len(bundle.steps):
                report.metric_mismatches.append(
                    f"num_steps: result={result.num_steps} "
                    f"recorded={len(bundle.steps)}"
                )
            if result.action_error_count != sum(
                1 for s in bundle.steps if s.action_error
            ):
                report.metric_mismatches.append(
                    f"action_error_count: result={result.action_error_count} "
                    f"recomputed="
                    f"{sum(1 for s in bundle.steps if s.action_error)}"
                )
            if result.verification_count != sum(
                1 for s in bundle.steps if s.verification_status is not None
            ):
                report.metric_mismatches.append(
                    f"verification_count: result={result.verification_count} "
                    f"recomputed="
                    f"{sum(1 for s in bundle.steps if s.verification_status is not None)}"
                )
            recovery_total = (
                result.recovery_success_count
                + result.recovery_failed_count
                + result.recovery_unresolved_count
            )
            if recovery_total != result.recovery_count:
                report.invariant_error_count += 1
                report.errors.append(
                    f"recovery outcome invariant violated: success("
                    f"{result.recovery_success_count}) + failed("
                    f"{result.recovery_failed_count}) + unresolved("
                    f"{result.recovery_unresolved_count}) = {recovery_total} "
                    f"!= recovery_count({result.recovery_count})"
                )
            replan_total = (
                result.replan_success_count
                + result.replan_failed_count
                + result.replan_unresolved_count
            )
            if replan_total != result.replan_count:
                report.invariant_error_count += 1
                report.errors.append(
                    f"replan outcome invariant violated: success("
                    f"{result.replan_success_count}) + failed("
                    f"{result.replan_failed_count}) + unresolved("
                    f"{result.replan_unresolved_count}) = {replan_total} "
                    f"!= replan_count({result.replan_count})"
                )

        # verification event completeness (R1): the event stream is the
        # authoritative verification record, so every verified StepRecord
        # must map to exactly one verification event with a consistent
        # summary — missing, duplicate or orphan events invalidate the trace
        events_by_step: dict[int, list[RuntimeEvent]] = {}
        for event in bundle.events:
            if event.event_type != RuntimeEventType.VERIFICATION:
                continue
            if event.step_index is None:
                report.errors.append(
                    f"verification event {event.event_id} has no step_index"
                )
                continue
            events_by_step.setdefault(event.step_index, []).append(event)
        for step in bundle.steps:
            step_events = events_by_step.get(step.step_index, [])
            if step.verification_status is None:
                if step_events:
                    report.errors.append(
                        f"orphan verification event(s) for step "
                        f"{step.step_index}: StepRecord.verification_status "
                        f"is None"
                    )
                continue
            if not step_events:
                report.errors.append(
                    f"missing verification event for verified step "
                    f"{step.step_index} (status={step.verification_status})"
                )
                continue
            if len(step_events) > 1:
                report.errors.append(
                    f"duplicate verification events for step "
                    f"{step.step_index}: {len(step_events)} events"
                )
                continue
            event = step_events[0]
            if event.outcome != step.verification_status:
                report.errors.append(
                    f"verification summary mismatch for step "
                    f"{step.step_index}: event outcome="
                    f"{event.outcome!r} != StepRecord."
                    f"verification_status={step.verification_status!r}"
                )
            event_kinds = sorted(
                {
                    str(sig.get("kind") if isinstance(sig, dict) else sig.kind.value)
                    for sig in event.data.get("signals", [])
                }
            )
            if event_kinds != sorted(step.failure_kinds):
                report.errors.append(
                    f"verification kinds mismatch for step "
                    f"{step.step_index}: event={event_kinds} != "
                    f"StepRecord.failure_kinds={sorted(step.failure_kinds)}"
                )

    # -- semantic verification replay ----------------------------------------

    def _semantic_verification_replay(
        self, bundle: TraceBundle, report: ReplayReport
    ) -> None:
        if bundle.trace_schema_version == 1:
            report.semantic_replay_skipped = True
            report.notes.append(
                "semantic replay skipped: legacy trace schema v1 has no "
                "structured Observation JSON artifacts"
            )
            return
        if bundle.trace_schema_version != 2:
            report.semantic_replay_skipped = True
            report.notes.append(
                f"semantic replay skipped: trace schema version "
                f"{bundle.trace_schema_version} is not supported for "
                f"semantic replay"
            )
            return
        if self._verifier is not None:
            # explicitly injected verifier (programmatic use / tests)
            verifier, spec_error = self._verifier, None
        else:
            verifier, spec_error = self._verifier_from_spec(bundle.manifest)
        if spec_error is not None:
            # the manifest HAS a verification_spec but it cannot be safely
            # rebuilt: fail closed (never replay against guessed defaults)
            report.errors.append(spec_error)
            return
        if verifier is None:
            report.semantic_replay_skipped = True
            if any(
                e.event_type == RuntimeEventType.VERIFICATION
                for e in bundle.events
            ):
                report.notes.append(
                    "semantic replay skipped: verification events present "
                    "but the manifest has no verifier spec — refusing to "
                    "replay against default settings"
                )
            return
        task = self._task_of(bundle)
        if task is None:
            report.semantic_replay_skipped = True
            report.notes.append(
                "semantic replay skipped: no TaskSpec recoverable from "
                "result.json or manifest"
            )
            return

        verification_events = {
            e.step_index: e
            for e in bundle.events
            if e.event_type == RuntimeEventType.VERIFICATION
            and e.step_index is not None
        }
        # fresh bookkeeping state: the verifier maintains it deterministically
        # from the recorded steps, exactly as it did during the live run
        reliability_state = ReliabilityState()
        for step in bundle.steps:
            event = verification_events.get(step.step_index)
            if event is None:
                continue
            missing_inputs = [
                field
                for field, value in (
                    ("observation_json_ref", step.observation_json_ref),
                    ("next_observation_json_ref", step.next_observation_json_ref),
                    ("action", step.action),
                )
                if not value
            ]
            if missing_inputs:
                # for a v2 verified step these are REQUIRED semantic replay
                # inputs: their absence is a replay error, not a skip note
                report.errors.append(
                    f"semantic replay cannot run for step "
                    f"{step.step_index}: missing required input(s): "
                    f"{', '.join(missing_inputs)}"
                )
                continue
            try:
                pre_obs = self._loader.read_observation(
                    bundle, step.observation_json_ref
                )
                post_obs = self._loader.read_observation(
                    bundle, step.next_observation_json_ref
                )
            except (OSError, ValidationError, ValueError) as exc:
                report.errors.append(
                    f"semantic replay failed for step {step.step_index}: "
                    f"cannot load structured observation: {exc}"
                )
                continue
            env_step = EnvironmentStep(
                observation=post_obs,
                reward=step.reward if step.reward is not None else 0.0,
                terminated=step.terminated,
                truncated=step.truncated,
                action_error=step.action_error,
            )
            verification = verifier.verify(
                task=task,
                pre_observation=pre_obs,
                action=step.action,
                env_step=env_step,
                history=bundle.steps[: step.step_index],
                reliability_state=reliability_state,
            )
            report.replayed_verifications += 1
            mismatch = self._compare_verification(step.step_index, event, verification)
            if mismatch:
                report.verification_mismatches.append(mismatch)

    @staticmethod
    def _verifier_from_spec(manifest: dict) -> tuple[object, str | None]:
        """Resolve the recorded verifier spec from manifest provenance.

        Returns (verifier, error):

        - (None, None): no verification_spec in the manifest — the caller
          decides (semantic replay skips with a note for verified traces);
        - (verifier, None): spec strictly valid for the standard built-in
          composition;
        - (None, error): the manifest HAS a spec but it is malformed, uses
          wrong types, misses required fields, or names an implementation
          that cannot be rebuilt offline — never guessed, never coerced.
        """
        if "verification_spec" not in manifest:
            return None, None
        spec = manifest.get("verification_spec")
        if spec is None:
            return None, None
        if not isinstance(spec, dict):
            return (
                None,
                f"verification_spec malformed: expected an object, got "
                f"{type(spec).__name__}",
            )
        try:
            parsed = VerifierSpec.model_validate(spec, strict=True)
        except ValidationError as exc:
            return (
                None,
                f"verification_spec malformed (fail-closed, no defaults "
                f"applied): {exc.errors(include_url=False)}",
            )
        if parsed.implementation != REBUILDABLE_VERIFIER_IMPLEMENTATION:
            return (
                None,
                f"verification_spec implementation {parsed.implementation!r} "
                f"is not rebuildable offline (supported: "
                f"{REBUILDABLE_VERIFIER_IMPLEMENTATION!r} with the standard "
                f"detector composition); semantic replay refuses to guess",
            )
        from web_harness.reliability.verifier import DefaultStepVerifier

        return (
            DefaultStepVerifier(
                detect_no_progress=parsed.detect_no_progress,
                detect_loop=parsed.detect_loop,
                loop_consecutive_threshold=parsed.loop_consecutive_threshold,
            ),
            None,
        )

    @staticmethod
    def _task_of(bundle: TraceBundle) -> TaskSpec | None:
        if bundle.result is not None:
            return bundle.result.task_spec
        manifest = bundle.manifest
        if manifest.get("benchmark") and manifest.get("task_id"):
            return TaskSpec(
                benchmark=str(manifest["benchmark"]),
                task_id=str(manifest["task_id"]),
                seed=manifest.get("seed"),
                max_steps=int(manifest.get("max_steps", 20)),
            )
        return None

    @staticmethod
    def _compare_verification(
        step_index: int, event: RuntimeEvent, verification
    ) -> str | None:
        """Deterministic comparison of a replayed verification pass.

        Compares status, the (kind, severity, signature) signal multiset and
        the deterministic fingerprint provenance (pre/post fingerprint,
        state_changed). `evidence` dicts are deliberately not compared: they
        carry the full detector context, and the semantic contract is the
        signal identity.
        """

        def signal_key(signal) -> tuple[str, str, str]:
            # recorded event signals are JSON dicts; replayed signals are
            # FailureSignal models
            if isinstance(signal, dict):
                return (
                    str(signal.get("kind")),
                    str(signal.get("severity")),
                    str(signal.get("signature")),
                )
            return (signal.kind.value, signal.severity.value, signal.signature)

        recorded_keys = sorted(
            signal_key(sig) for sig in event.data.get("signals", [])
        )
        replayed_keys = sorted(
            signal_key(sig) for sig in verification.signals
        )
        recorded_status = event.outcome
        replayed_status = verification.status.value
        mismatches = []
        if recorded_status != replayed_status:
            mismatches.append(
                f"status recorded={recorded_status!r} != "
                f"replayed={replayed_status!r}"
            )
        if recorded_keys != replayed_keys:
            mismatches.append(
                f"signals recorded={recorded_keys} != replayed={replayed_keys}"
            )
        for field in ("pre_fingerprint", "post_fingerprint", "state_changed"):
            recorded_value = event.data.get(field)
            replayed_value = getattr(verification, field)
            if recorded_value != replayed_value:
                mismatches.append(
                    f"{field} recorded={recorded_value!r} != "
                    f"replayed={replayed_value!r}"
                )
        if not mismatches:
            return None
        return f"step {step_index}: " + "; ".join(mismatches)
