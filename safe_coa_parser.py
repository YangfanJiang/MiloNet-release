"""Safe variants of Chain-of-Abstraction parsing utilities."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from llama_index.agent.coa.output_parser import ChainOfAbstractionParser
from llama_index.core.tools import AsyncBaseTool, ToolOutput
import helper


@dataclass
class _ConcatPart:
    kind: str  # "text" or "placeholder"
    value: str
    leading_spaces: int = 0
    trailing_spaces: int = 0


ConcatToken = Dict[str, Union[str, List[_ConcatPart]]]
TokenType = Union[str, ConcatToken]


class SafeChainOfAbstractionParser(ChainOfAbstractionParser):
    """Chain-of-Abstraction parser resilient to minor formatting glitches.

    Some LLM generations occasionally emit tool calls such as
    `[FUNC document_retriever_tool("where was", y1, "formed") = y2]`, which is
    invalid JSON for the default parser.  This wrapper sanitises the raw input
    list by stitching contiguous text/placeholder fragments into a single
    concatenated payload so tool execution can proceed.
    
    This version RAISES an error on duplicate placeholders (for UI.py CoA agents 
    that may have placeholder references).
    """
    
    def __init__(self, verbose: bool = False, auto_fix_duplicates: bool = False):
        super().__init__(verbose=verbose)
        self._auto_fix_duplicates = auto_fix_duplicates

    def _sanitize_inputs(self, raw_inputs: str) -> List[TokenType]:
        """Parse a raw input list into JSON or a concat descriptor."""
        try:
            return json.loads("[" + raw_inputs + "]")
        except json.JSONDecodeError:
            tokens = self._tokenize_fragments(raw_inputs)
            if not tokens:
                return []

            # If the malformed call only contains a single token, treat it as a
            # plain string literal (e.g., missing surrounding quotes).
            if len(tokens) == 1:
                token = tokens[0]
                stripped = token.strip()
                if stripped.startswith("\"") and stripped.endswith("\""):
                    return [json.loads(stripped)]
                return [stripped]

            parts: List[_ConcatPart] = []
            for token in tokens:
                leading = len(token) - len(token.lstrip(" "))
                trailing = len(token) - len(token.rstrip(" "))
                stripped = token.strip()
                if not stripped:
                    continue
                if stripped.startswith("\"") and stripped.endswith("\""):
                    text = json.loads(stripped)
                    parts.append(
                        _ConcatPart(
                            kind="text",
                            value=text,
                            leading_spaces=leading,
                            trailing_spaces=trailing,
                        )
                    )
                else:
                    parts.append(
                        _ConcatPart(
                            kind="placeholder",
                            value=stripped,
                            leading_spaces=leading,
                            trailing_spaces=trailing,
                        )
                    )

            return [
                {
                    "type": "concat",
                    "parts": parts,
                }
            ]

    def _tokenize_fragments(self, raw_inputs: str) -> List[str]:
        pattern = r'"(?:\\.|[^"\\])*"|[^,]+'
        return [frag for frag in re.findall(pattern, raw_inputs) if frag]

    def _resolve_token(self, token: TokenType, results: Dict[str, str]) -> str:
        if isinstance(token, dict) and token.get("type") == "concat":
            parts: Sequence[_ConcatPart] = token.get("parts", [])  # type: ignore
            fragments: List[str] = []
            for part in parts:
                base = (
                    results.get(part.value, part.value)
                    if part.kind == "placeholder"
                    else part.value
                )
                segment = (
                    " " * part.leading_spaces
                    + str(base)
                    + " " * part.trailing_spaces
                )
                fragments.append(segment)
            return "".join(fragments)
        if isinstance(token, str):
            return results.get(token, token)
        return str(token)

    async def aparse(
        self, solution: str, tools_by_name: Dict[str, AsyncBaseTool]
    ) -> Tuple[str, List[ToolOutput]]:  # type: ignore[override]
        import networkx as nx  # local import to avoid unused dependency warnings

        # Isolate the initial plan block; everything after execution logs should be left intact.
        plan_break_markers = [
            "==== Executing",
            "=== Calling Function",
            "==== Running",
            "==== Tool",
        ]
        plan_end_index = len(solution)
        for marker in plan_break_markers:
            idx = solution.find(marker)
            if idx != -1:
                plan_end_index = min(plan_end_index, idx)

        plan_section = solution[:plan_end_index]
        trailing_section = solution[plan_end_index:]

        func_calls = re.findall(r"\[FUNC (\w+)\((.*?)\) = (\w+)\]", plan_section)

        # Attempt to auto-correct common formatting slips before enforcing errors.
        bracket_pattern = re.compile(r"\[([^\]]+)\]")
        corrected_plan = plan_section
        corrections_made = False
        unfixable_missing_func: List[str] = []

        for raw_call in bracket_pattern.findall(plan_section):
            inner = raw_call.strip()
            if "=" not in inner:
                continue
            if inner.startswith("FUNC "):
                continue

            match = re.match(r"(\w+)\((.*?)\)\s*=\s*(y\d+)\s*$", inner)
            if match:
                func_name, args_raw, placeholder = match.groups()
                replacement = f"FUNC {func_name}({args_raw}) = {placeholder}"
                corrected_plan = corrected_plan.replace(
                    f"[{raw_call}]", f"[{replacement}]", 1
                )
                corrections_made = True
            else:
                unfixable_missing_func.append(inner)

        if corrections_made:
            plan_section = corrected_plan
            func_calls = re.findall(r"\[FUNC (\w+)\((.*?)\) = (\w+)\]", plan_section)

        placeholder_issues: List[str] = []
        for raw_call in bracket_pattern.findall(plan_section):
            inner = raw_call.strip()
            if "=" not in inner:
                continue
            if not inner.startswith("FUNC "):
                # Either we failed to auto-correct or the pattern is fundamentally invalid.
                unfixable_missing_func.append(inner)
                continue
            if not re.search(r"= y\d+\s*$", inner):
                placeholder_issues.append(inner)

        if unfixable_missing_func or placeholder_issues:
            issues: List[str] = []
            if unfixable_missing_func:
                examples = ", ".join(f"[{entry}]" for entry in unfixable_missing_func[:3])
                issues.append(
                    "Missing `FUNC` keyword before tool name and cannot auto-correct. Examples: "
                    + examples
                )
            if placeholder_issues:
                examples = ", ".join(f"[{entry}]" for entry in placeholder_issues[:3])
                issues.append(
                    "Each call must terminate with `= yX`. Examples: " + examples
                )
            raise ValueError(
                "CRITICAL FORMAT ERROR: Chain-of-Abstraction plan uses an invalid "
                "function-call format.\n"
                "Required pattern: [FUNC actual_tool_name(\"query\") = yX].\n"
                + "\n".join(issues)
            )

        if self._auto_fix_duplicates:
            # Step 1: remove duplicate calls with same function name and inputs
            seen_signatures = set()
            unique_calls = []
            for func_name, inputs_raw, output in func_calls:
                signature = (func_name, inputs_raw.strip())
                if signature in seen_signatures:
                    if self._verbose:
                        print(f"Skipping duplicate function call [FUNC {func_name}({inputs_raw}) = {output}]")
                    continue
                seen_signatures.add(signature)
                unique_calls.append((func_name, inputs_raw, output))

            # Step 2: reassign placeholders sequentially, aborting on ambiguous duplicates.
            placeholder_map: Dict[str, str] = {}
            counter = 1
            corrected_plan = plan_section
            for func_name, inputs_raw, output in unique_calls:
                if output in placeholder_map:
                    raise ValueError(
                        "CRITICAL FORMAT ERROR: duplicate placeholder usage detected.\n"
                        "Cannot auto-correct because placeholder is reused across distinct function calls.\n"
                        f"Offending call: [FUNC {func_name}({inputs_raw}) = {output}]"
                    )

                new_output = f"y{counter}"
                counter += 1
                placeholder_map[output] = new_output

                original_call = f"[FUNC {func_name}({inputs_raw}) = {output}]"
                corrected_call = f"[FUNC {func_name}({inputs_raw.strip()}) = {new_output}]"
                corrected_plan = corrected_plan.replace(original_call, corrected_call, 1)

            for old_placeholder, new_placeholder in placeholder_map.items():
                if old_placeholder == new_placeholder:
                    continue
                corrected_plan = re.sub(
                    rf"\b{re.escape(old_placeholder)}\b",
                    new_placeholder,
                    corrected_plan,
                )

            plan_section = corrected_plan
            func_calls = re.findall(r"\[FUNC (\w+)\((.*?)\) = (\w+)\]", plan_section)
        else:
            # Validate unique assignment targets (right side of =)
            assigned_placeholders = [output for _, _, output in func_calls]
            if len(assigned_placeholders) != len(set(assigned_placeholders)):
                duplicates = [p for p in set(assigned_placeholders) if assigned_placeholders.count(p) > 1]

                duplicate_details = []
                for fn, inp, out in func_calls:
                    if out in duplicates:
                        duplicate_details.append(f"[FUNC {fn}({inp[:50]}...) = {out}]")

                error_msg = (
                    f"CRITICAL ERROR: Duplicate placeholder assignments detected.\n"
                    f"Duplicated placeholders: {duplicates}\n"
                    f"Each function call MUST assign to a unique placeholder (y1, y2, y3, etc.).\n"
                    f"Problematic calls:\n" + "\n".join(duplicate_details) + "\n"
                    f"Please regenerate the plan with unique placeholders."
                )
                raise ValueError(error_msg)

        # Final HARD syntax validation per prompt contract.
        strict_pattern = re.compile(r'^\[FUNC [A-Za-z0-9_]+\("([^"\\]|\\.)*"\) = y[1-9]\d*\]$')
        invalid_lines: List[str] = []
        for raw_call in bracket_pattern.findall(plan_section):
            snippet = f"[{raw_call.strip()}]"
            if "=" not in snippet:
                continue
            if strict_pattern.fullmatch(snippet):
                continue

            call_match = re.match(r'^FUNC ([A-Za-z0-9_]+)\((.*?)\) = (y[1-9]\d*)$', raw_call.strip())
            if call_match:
                func_name, args_raw, placeholder = call_match.groups()
                canonical_args = args_raw.strip()
                canonical_call = f"[FUNC {func_name}({canonical_args}) = {placeholder}]"
                if strict_pattern.fullmatch(canonical_call):
                    plan_section = plan_section.replace(snippet, canonical_call, 1)
                    continue

            invalid_lines.append(snippet)

        if invalid_lines:
            examples = ", ".join(invalid_lines[:3])
            raise ValueError(
                "CRITICAL FORMAT ERROR: Chain-of-Abstraction plan lines must match the strict pattern "
                "`[FUNC tool_name(\"query string\") = yN]`.\n"
                f"Example failures: {examples}"
            )

        # Reassemble the solution with the potentially corrected plan section.
        solution = plan_section + trailing_section

        graph = nx.DiGraph()
        for func_name, inputs_raw, output in func_calls:
            sanitized_inputs = self._sanitize_inputs(inputs_raw)
            graph.add_node(
                output,
                func_name=func_name,
                inputs=sanitized_inputs,
            )
            for token in sanitized_inputs:
                if isinstance(token, dict) and token.get("type") == "concat":
                    for part in token["parts"]:  # type: ignore[index]
                        if part.kind == "placeholder":
                            graph.add_edge(part.value, output)
                elif isinstance(token, str):
                    graph.add_edge(token, output)

        execution_levels = defaultdict(list)
        for node in nx.topological_sort(graph):
            level = (
                max(
                    [execution_levels[pred] for pred in graph.predecessors(node)],
                    default=-1,
                )
                + 1
            )
            execution_levels[node] = level

        level_groups = defaultdict(list)
        for node, level in execution_levels.items():
            level_groups[level].append(node)

        results: Dict[str, str] = {}
        tool_outputs: List[ToolOutput] = []
        graph_nodes = {node[0]: node[1] for node in graph.nodes(data=True)}

        for level in sorted(level_groups.keys()):
            level_nodes = level_groups[level]
            parallel_results: Dict[str, str] = {}
            for placeholder in level_nodes:
                node_data = graph_nodes.get(placeholder, {})
                if not node_data:
                    continue

                func_name = node_data.get("func_name")
                inputs_tokens = node_data.get("inputs", [])
                input_values = [self._resolve_token(tok, results) for tok in inputs_tokens]

                if self._verbose:
                    print(
                        f"==== Executing {func_name} with inputs {input_values} ====",
                        flush=True,
                    )

                try:
                    raw_tool_output = await tools_by_name[func_name].acall(
                        *input_values
                    )
                    tool_outputs.append(
                        ToolOutput(
                            content=str(raw_tool_output),
                            tool_name=func_name,
                            raw_output=raw_tool_output,
                            raw_input={"args": input_values},
                            is_error=False,
                        )
                    )
                except Exception as e:  # pragma: no cover - safety net
                    tool_outputs.append(
                        ToolOutput(
                            content=str(e),
                            tool_name=func_name,
                            raw_output=None,
                            raw_input={"args": input_values},
                            is_error=True,
                        )
                    )
                    break

                parallel_results[placeholder] = str(raw_tool_output)
            results.update(parallel_results)

        for placeholder, value in results.items():
            solution = solution.replace(f"{placeholder}", '"' + str(value) + '"')

        helper.last_plan_text = solution
        return solution, tool_outputs