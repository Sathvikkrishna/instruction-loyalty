"""Run the research agent under ILS evaluation, then report and chart the result."""

from __future__ import annotations

import argparse
import sys
import textwrap

import anthropic

import config
from agent import AgentStep, build_agent, iter_steps
from config import CRITICAL_KEYS, INSTRUCTIONS, INSTRUCTIONS_BY_KEY, Verdict
from evaluator import (
    BreakerTrip, ILSEvaluator, ILSReport, StepEvaluation, build_report,
    load_report, save_report,
)

# --- console helpers --------------------------------------------------------
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, YELLOW, BLUE = "\033[32m", "\033[31m", "\033[33m", "\033[34m"

VERDICT_STYLE = {
    Verdict.COMPLIANT: (GREEN, "COMPLIANT"),
    Verdict.VIOLATED: (RED, "VIOLATED "),
    Verdict.NOT_APPLICABLE: (DIM, "N/A      "),
}
WIDTH = 96


def rule(char: str = "-") -> str:
    return char * WIDTH


def header(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}\n{rule('=')}")


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def bar(fraction: float | None, width: int = 20) -> str:
    if fraction is None:
        return DIM + "-" * width + RESET
    filled = round(fraction * width)
    colour = GREEN if fraction >= 0.85 else YELLOW if fraction >= config.BREAKER_THRESHOLD else RED
    return f"{colour}{'#' * filled}{RESET}{DIM}{'.' * (width - filled)}{RESET}"


# --- per-step output --------------------------------------------------------
def print_step(step: AgentStep) -> None:
    print(f"\n{rule()}")
    print(f"{BOLD}STEP {step.index}{RESET}  "
          f"{DIM}{len(step.tool_calls)} tool call(s){RESET}")
    print(rule())
    if step.text:
        print(wrap(step.text[:900] + ("..." if len(step.text) > 900 else ""), "  "))
    else:
        print(f"{DIM}  (no visible text){RESET}")
    for call in step.tool_calls:
        print(f"  {BLUE}-> {call.name}{RESET}: {call.query[:110]}")


def print_evaluation(evaluation: StepEvaluation, evaluator: ILSEvaluator) -> None:
    print(f"\n  {BOLD}ILS judge{RESET}")
    for ins in INSTRUCTIONS:
        verdict = evaluation.verdicts.get(ins.key, Verdict.NOT_APPLICABLE)
        colour, tag = VERDICT_STYLE[verdict]
        rolling = evaluator.rolling_compliance(ins.key)
        rolling_txt = f"{DIM}rolling {100 * rolling:3.0f}%{RESET}" if rolling is not None else f"{DIM}rolling   - {RESET}"
        print(f"    {colour}{tag}{RESET} {ins.key:<24} {rolling_txt}")
        reason = evaluation.reasons.get(ins.key, "no verdict returned")
        print(f"      {DIM}{textwrap.shorten(reason, 82)}{RESET}")


# --- circuit breaker --------------------------------------------------------
def fire_breaker(
    step_index: int, trip: BreakerTrip, evaluator: ILSEvaluator, policy: str = "ask"
) -> bool:
    """Warn, then decide whether to keep going.

    `policy` is "ask" (pause for the operator), "continue", or "abort" - the
    latter two make unattended runs deterministic instead of depending on
    whether a tty happens to be attached.
    """
    critical = INSTRUCTIONS_BY_KEY[trip.key]
    print(f"\n{RED}{BOLD}{'!' * WIDTH}{RESET}")
    print(f"{RED}{BOLD}  CIRCUIT BREAKER - {trip.key} DEGRADED{RESET}")
    print(f"{RED}{'!' * WIDTH}{RESET}")
    print(f"  Instruction : {BOLD}{trip.key}{RESET} (severity {critical.severity.value})")
    print(f"  Watching    : {DIM}{', '.join(CRITICAL_KEYS)}{RESET}")
    print(f"  Rolling     : {RED}{BOLD}{100 * trip.rolling:.0f}%{RESET} over the last "
          f"{trip.samples} applicable step(s) "
          f"{DIM}(window {config.ROLLING_WINDOW}){RESET}")
    print(f"  Threshold   : {100 * config.BREAKER_THRESHOLD:.0f}%")
    print(f"  Tripped at  : step {step_index}")
    print()
    print(wrap(
        f"A Critical rule has collapsed: {critical.directive} Anything the agent "
        "produces past this point should be treated as unverified until the "
        "trajectory is reviewed.", "  "
    ))
    if policy == "continue":
        print(f"{YELLOW}  --on-breaker=continue: continuing under a tripped breaker.{RESET}")
        return True
    if policy == "abort":
        print(f"{RED}  --on-breaker=abort: stopping the run.{RESET}")
        return False

    if not sys.stdin.isatty():
        print(f"{DIM}  no tty and --on-breaker=ask - aborting. "
              f"Pass --on-breaker=continue for unattended runs.{RESET}")
        return False

    print(f"\n  {BOLD}[c]{RESET} continue the run   {BOLD}[a]{RESET} abort and report on what ran so far")

    while True:
        try:
            choice = input(f"  {BOLD}>{RESET} ").strip().lower()
        except EOFError:
            print(f"{DIM}  no tty - aborting{RESET}")
            return False
        if choice in ("c", "continue", ""):
            print(f"{YELLOW}  Continuing under a tripped breaker.{RESET}")
            return True
        if choice in ("a", "abort", "q"):
            print(f"{RED}  Aborting the run.{RESET}")
            return False
        print(f"{DIM}  Enter 'c' or 'a'.{RESET}")


# --- final report -----------------------------------------------------------
def print_report(report: ILSReport) -> None:
    header("ILS REPORT")
    print(f"  Task        : {textwrap.shorten(report.task or config.RESEARCH_TASK, WIDTH - 16)}")
    print(f"  Agent model : {config.AGENT_MODEL}")
    print(f"  Judge model : {config.JUDGE_MODEL}")
    print(f"  Steps judged: {report.steps_run}"
          + (f"   {RED}(run aborted by circuit breaker){RESET}" if report.aborted else ""))

    print(f"\n{BOLD}  Per-instruction compliance{RESET}")
    print(f"  {'Instruction':<24}{'Sev':<10}{'Compliance':<24}{'n':<5}{'C/V/NA':<12}"
          f"{'1st viol.':<11}{'Recovery'}")
    print(f"  {rule('-')[:WIDTH - 2]}")
    for metric in report.metrics:
        rate = metric.compliance_rate
        rate_txt = "  n/a" if rate is None else f"{100 * rate:3.0f}%"
        if metric.is_thin:
            rate_txt += f"{YELLOW}*{RESET}"
        else:
            rate_txt += " "
        first = "none" if metric.first_violation_step is None else f"step {metric.first_violation_step}"
        recovery = metric.recovery_rate
        recovery_txt = "n/a" if recovery is None else f"{100 * recovery:.0f}%"
        counts = f"{metric.compliant}/{metric.violated}/{metric.not_applicable}"
        print(f"  {metric.label:<24}{metric.severity:<10}{bar(rate)} {rate_txt}  "
              f"{metric.applicable:<5}{counts:<12}{first:<11}{recovery_txt}")

    thin = report.thin_metrics
    if thin:
        names = ", ".join(m.label for m in thin)
        print(f"\n  {YELLOW}* {names}: fewer than {config.MIN_CONFIDENT_SAMPLES} "
              f"applicable steps - rate is not yet meaningful.{RESET}")

    print(f"\n{BOLD}  Violations in order{RESET}")
    violations = [
        (ev.step_index, key, ev.reasons[key])
        for ev in report.evaluations
        for key, verdict in ev.verdicts.items()
        if verdict is Verdict.VIOLATED
    ]
    if violations:
        for step_index, key, reason in violations:
            severity = INSTRUCTIONS_BY_KEY[key].severity.value
            print(f"    {RED}step {step_index:<3}{RESET} {key} {DIM}({severity}){RESET}")
            print(f"      {DIM}{textwrap.shorten(reason, WIDTH - 10)}{RESET}")
    else:
        print(f"    {GREEN}none - every applicable criterion held{RESET}")

    if report.breaker_events:
        print(f"\n{BOLD}  Circuit-breaker events{RESET}")
        for event in report.breaker_events:
            action = "continued" if event["continued"] else "aborted"
            name = event.get("instruction", "?")
            samples = event.get("samples")
            basis = f" over {samples} step(s)" if samples else ""
            print(f"    {RED}step {event['step']}{RESET}: {BOLD}{name}{RESET} rolling "
                  f"{100 * event['rolling']:.0f}% < {100 * config.BREAKER_THRESHOLD:.0f}%"
                  f"{basis} -> operator {action}")

    total_in = (report.agent_input_tokens + report.judge_input_tokens
                + report.agent_cache_read_tokens + report.agent_cache_write_tokens)
    total_out = report.agent_output_tokens + report.judge_output_tokens
    if total_in or total_out:
        print(f"\n{BOLD}  Run cost{RESET}")
        print(f"    agent  {report.agent_input_tokens:>9,} in  "
              f"{report.agent_output_tokens:>7,} out")
        if report.agent_cache_read_tokens or report.agent_cache_write_tokens:
            print(f"    cache  {report.agent_cache_read_tokens:>9,} read "
                  f"{report.agent_cache_write_tokens:>7,} written  "
                  f"{DIM}(read at {config.CACHE_READ_MULTIPLIER:.0%} of input price)"
                  f"{RESET}")
        elif config.PROMPT_CACHING:
            print(f"    {YELLOW}cache  caching on, but no cache hits recorded{RESET}")
        print(f"    judge  {report.judge_input_tokens:>9,} in  "
              f"{report.judge_output_tokens:>7,} out")
        cost = report.estimated_cost_usd
        if cost is not None:
            print(f"    {BOLD}estimated ${cost:.3f}{RESET} "
                  f"{DIM}({total_in + total_out:,} tokens){RESET}")

    score = report.overall_ils
    print(f"\n{rule('=')}")
    if score is None:
        print(f"  {BOLD}OVERALL ILS SCORE: n/a{RESET} {DIM}(no applicable verdicts){RESET}")
    else:
        colour = GREEN if score >= 85 else YELLOW if score >= 70 else RED
        band = "healthy" if score >= 85 else "degraded" if score >= 70 else "at risk" if score >= 50 else "failing"
        print(f"  {BOLD}OVERALL ILS SCORE: {colour}{score:.1f}/100{RESET}  "
              f"{DIM}({band}; severity-weighted){RESET}")
        if report.unweighted_ils is not None:
            print(f"  {BOLD}UNWEIGHTED MEAN:   {report.unweighted_ils:.1f}/100{RESET} "
                  f"{DIM}(severity-independent; comparable across weightings){RESET}")
        if report.thin_metrics:
            print(f"  {YELLOW}caution: {len(report.thin_metrics)} of "
                  f"{len(report.metrics)} instructions scored on fewer than "
                  f"{config.MIN_CONFIDENT_SAMPLES} applicable steps.{RESET}")
    print(rule("="))


# --- error reporting --------------------------------------------------------
def _root_api_error(exc: BaseException) -> anthropic.APIError | None:
    """Walk the cause chain - LangChain rewraps the SDK's exceptions."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, anthropic.APIError):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def explain_api_error(exc: BaseException) -> str:
    """One actionable line instead of a sixty-line traceback."""
    api_error = _root_api_error(exc)
    if api_error is None:
        return f"{type(exc).__name__}: {exc}"

    detail = getattr(api_error, "message", None) or str(api_error)
    if isinstance(api_error, anthropic.AuthenticationError):
        return f"Anthropic rejected the API key: {detail}"
    if isinstance(api_error, anthropic.RateLimitError):
        return f"Rate limited by Anthropic: {detail}"
    if isinstance(api_error, anthropic.APIConnectionError):
        return f"Could not reach the Anthropic API: {detail}"
    if isinstance(api_error, anthropic.BadRequestError):
        if "credit balance" in detail.lower():
            return (
                "Anthropic API credit balance is too low. Add credits at "
                "console.anthropic.com -> Plans & Billing, then re-run."
            )
        return f"Anthropic rejected the request: {detail}"
    return f"Anthropic API error: {detail}"


# --- main -------------------------------------------------------------------
def run(task: str, max_steps: int, on_breaker: str = "ask") -> ILSReport:
    agent = build_agent()
    evaluator = ILSEvaluator()
    breaker_events: list[dict] = []
    aborted = False

    header(f"ILS AGENT RUN  ·  {config.AGENT_MODEL}  ·  max {max_steps} steps")
    print(wrap(task, "  "))

    agent_input = agent_output = cache_read = cache_write = 0
    steps = iter_steps(agent, task, max_steps)
    for step in steps:
        agent_input += step.input_tokens
        agent_output += step.output_tokens
        cache_read += step.cache_read_tokens
        cache_write += step.cache_write_tokens
        print_step(step)
        evaluation = evaluator.evaluate(step)
        print_evaluation(evaluation, evaluator)

        spent = config.estimate_cost(
            config.AGENT_MODEL, agent_input, agent_output,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        )
        judged = config.estimate_cost(
            config.JUDGE_MODEL, evaluator.judge_input_tokens, evaluator.judge_output_tokens
        )
        if spent is not None and judged is not None:
            print(f"    {DIM}running cost: ${spent + judged:.3f}{RESET}")

        trip = evaluator.breaker_tripped()
        if trip is not None:
            should_continue = fire_breaker(step.index, trip, evaluator, on_breaker)
            breaker_events.append(
                {
                    "step": step.index,
                    "instruction": trip.key,
                    "rolling": trip.rolling,
                    "samples": trip.samples,
                    "continued": should_continue,
                }
            )
            if not should_continue:
                aborted = True
                steps.close()
                break

    return build_report(
        evaluator.evaluations, breaker_events, aborted, task=task,
        usage={
            "agent_input_tokens": agent_input,
            "agent_output_tokens": agent_output,
            "agent_cache_read_tokens": cache_read,
            "agent_cache_write_tokens": cache_write,
            "judge_input_tokens": evaluator.judge_input_tokens,
            "judge_output_tokens": evaluator.judge_output_tokens,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ILS-evaluated research agent.")
    parser.add_argument("--task", default=config.RESEARCH_TASK, help="research task")
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--no-charts", action="store_true", help="skip chart rendering")
    parser.add_argument(
        "--on-breaker", choices=["ask", "continue", "abort"], default="ask",
        help="what to do when the circuit breaker trips (default: ask)",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="re-render from the run file instead of running the agent (no API calls)",
    )
    parser.add_argument(
        "--run-file", default="run.json",
        help="filename under output/ to save to or replay from (default: run.json)",
    )
    args = parser.parse_args()

    run_path = config.OUTPUT_DIR / args.run_file
    # Charts inherit the run file's name, so separate runs never overwrite
    # each other's output.
    suffix = "" if run_path.stem == "run" else f"_{run_path.stem.removeprefix('run_')}"
    if args.replay:
        if not run_path.exists():
            print(f"{RED}No saved run at {run_path}.{RESET}")
            return 1
        report = load_report(run_path)
        print_report(report)
        import visualizer

        header("CHARTS")
        for path in visualizer.render_all(report, suffix):
            print(f"  wrote {path}")
        return 0

    missing = config.validate_env()
    if missing:
        print(f"{RED}Missing required key(s): {', '.join(missing)}{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill them in.{RESET}")
        return 1

    try:
        report = run(args.task, args.max_steps, args.on_breaker)
    except Exception as exc:  # surface one actionable line, not a traceback
        print(f"\n{RED}Run failed.{RESET} {explain_api_error(exc)}")
        return 1

    if not report.evaluations:
        print(f"\n{YELLOW}No steps were judged - nothing to report.{RESET}")
        return 1

    print_report(report)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_report(report, run_path)
    print(f"\n{DIM}  saved raw verdicts to {run_path} (re-render with --replay){RESET}")

    if not args.no_charts:
        import visualizer  # imported late so --no-charts needs no matplotlib

        header("CHARTS")
        for path in visualizer.render_all(report, suffix):
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
