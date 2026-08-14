from __future__ import annotations

from app.core.stream_events import event_print as print
import json
import operator
import re
import time
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.core.llm_client import client, model_name
from config import config


class ComplexRAGState(TypedDict, total=False):
    original_question: str
    is_complex: bool
    plan_reason: str
    subquestions: list[str]

    evidence_parts: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    merged_evidence: str
    merged_sufficient: bool
    grade_reason: str
    final_context: str


class WorkerState(TypedDict):
    index: int
    subquestion: str


class ComplexRAGController:

    COMPLEX_HINTS = (
        "比较",
        "对比",
        "区别",
        "优缺点",
        "分别",
        "以及",
        "同时",
        "之间的关系",
        "协同",
        "各自",
        "多个方面",
        "完整流程",
        "整体架构",
        "综合分析",
    )

    def __init__(
        self,
        raw_retrieve: Callable[[str], str],
        simple_retrieve: Callable[[str], str],
    ):
        self.raw_retrieve = raw_retrieve
        self.simple_retrieve = simple_retrieve

        cfg = config.get(
            "complex_rag",
            {},
        )

        self.enabled = bool(
            cfg.get(
                "enabled",
                True,
            )
        )

        configured_model = str(
            cfg.get(
                "model",
                "",
            )
            or ""
        ).strip()

        self.model = (
            configured_model
            or model_name
        )

        self.max_subquestions = max(
            2,
            min(
                4,
                int(
                    cfg.get(
                        "max_subquestions",
                        3,
                    )
                ),
            ),
        )

        self.max_each_result_chars = max(
            1500,
            int(
                cfg.get(
                    "max_each_result_chars",
                    4500,
                )
            ),
        )

        self.max_merged_chars = max(
            5000,
            int(
                cfg.get(
                    "max_merged_chars",
                    14000,
                )
            ),
        )

        self.simple_fast_path_max_chars = max(
            10,
            int(
                cfg.get(
                    "simple_fast_path_max_chars",
                    32,
                )
            ),
        )

        self.max_attempts = max(
            1,
            int(
                cfg.get(
                    "max_attempts",
                    2,
                )
            ),
        )

        self.graph = self._build_graph()

    def _call_model(
        self,
        messages: list[dict],
    ) -> str:

        last_error = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            try:
                response = (
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                )

                return str(
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                ).strip()

            except Exception as exc:
                last_error = exc

                print(
                    "[ComplexRAG] "
                    f"模型调用失败 "
                    f"{attempt}/{self.max_attempts}："
                    f"{exc}"
                )

                if attempt < self.max_attempts:
                    time.sleep(attempt)

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Complex RAG 模型调用失败"
        )

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        text = str(
            text
            or ""
        ).strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            if not match:
                raise

            return json.loads(
                match.group(0)
            )

    def _is_obviously_simple(
        self,
        question: str,
    ) -> bool:

        question = str(
            question
            or ""
        ).strip()

        if (
            len(question)
            > self.simple_fast_path_max_chars
        ):
            return False

        return not any(
            hint in question
            for hint in self.COMPLEX_HINTS
        )

    def _plan_question(
        self,
        question: str,
    ) -> dict:

        messages = [
            {
                "role": "system",
                "content": (
                    "你是 RAG 系统中的问题规划器。"
                    "判断问题是否需要拆成多个独立检索子问题。\n\n"
                    "规则：\n"
                    "1. 单一事实、单一概念属于 simple；\n"
                    "2. 比较多个对象、多个方面、多个机制关系、"
                    "完整流程属于 complex；\n"
                    "3. complex 拆成 2 到 4 个可独立检索的子问题；\n"
                    "4. 子问题共同覆盖原问题；\n"
                    "5. 子问题不要重复；\n"
                    "6. 不回答问题。\n\n"
                    "只输出 JSON：\n"
                    "{\"complex\": true, "
                    "\"reason\": \"原因\", "
                    "\"subquestions\": [\"问题1\", \"问题2\"]}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        try:
            raw = self._call_model(messages)
            data = self._parse_json(raw)

            value = data.get(
                "complex",
                False,
            )

            if isinstance(value, str):
                is_complex = (
                    value.strip().lower()
                    in (
                        "true",
                        "1",
                        "yes",
                        "是",
                    )
                )
            else:
                is_complex = bool(value)

            raw_subquestions = data.get(
                "subquestions",
                [],
            )

            if not isinstance(
                raw_subquestions,
                list,
            ):
                raw_subquestions = []

            subquestions = []

            for item in raw_subquestions:
                item = str(
                    item
                    or ""
                ).strip()

                if (
                    not item
                    or item in subquestions
                ):
                    continue

                subquestions.append(item)

                if (
                    len(subquestions)
                    >= self.max_subquestions
                ):
                    break

            if (
                is_complex
                and len(subquestions) < 2
            ):
                is_complex = False
                subquestions = []

            return {
                "complex": is_complex,
                "reason": str(
                    data.get(
                        "reason",
                        "",
                    )
                ).strip(),
                "subquestions": subquestions,
            }

        except Exception as exc:
            print(
                "[ComplexRAG] "
                "复杂度规划失败，"
                "降级普通 Corrective RAG："
                f"{exc}"
            )

            return {
                "complex": False,
                "reason": "Planner 不可用，自动降级。",
                "subquestions": [],
            }

    def _grade_merged_evidence(
        self,
        question: str,
        evidence: str,
    ) -> tuple[bool, str]:

        evidence = str(
            evidence
            or ""
        )[:self.max_merged_chars]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是复杂 RAG 的证据覆盖度评判器。"
                    "判断多个子问题的检索结果合并后，"
                    "是否已经足以覆盖用户原始问题。\n\n"
                    "只要整体证据能支持回答主要部分即可判 true；"
                    "如果重要部分仍完全缺失则判 false。"
                    "不要回答问题。\n\n"
                    "只输出 JSON："
                    "{\"sufficient\": true, \"reason\": \"原因\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "【原始问题】\n"
                    f"{question}\n\n"
                    "【合并证据】\n"
                    f"{evidence}"
                ),
            },
        ]

        try:
            raw = self._call_model(messages)
            data = self._parse_json(raw)

            value = data.get(
                "sufficient",
                True,
            )

            if isinstance(value, str):
                sufficient = (
                    value.strip().lower()
                    in (
                        "true",
                        "1",
                        "yes",
                        "是",
                    )
                )
            else:
                sufficient = bool(value)

            return (
                sufficient,
                str(
                    data.get(
                        "reason",
                        "",
                    )
                ).strip(),
            )

        except Exception as exc:
            print(
                "[ComplexRAG] "
                "Coverage Grade 失败，"
                "保留合并证据："
                f"{exc}"
            )

            return (
                True,
                "Coverage Grade 不可用，已降级使用合并证据。",
            )

    def _plan_node(
        self,
        state: ComplexRAGState,
    ) -> dict:

        plan = self._plan_question(
            state["original_question"]
        )

        print()
        print(
            "[ComplexRAG] Planner："
            f"complex={plan['complex']} | "
            f"reason={plan['reason']}"
        )

        for index, subquestion in enumerate(
            plan["subquestions"],
            start=1,
        ):
            print(
                "[ComplexRAG] "
                f"子问题 {index}："
                f"{subquestion}"
            )

        return {
            "is_complex": plan["complex"],
            "plan_reason": plan["reason"],
            "subquestions": plan["subquestions"],
        }

    def _route_after_plan(
        self,
        state: ComplexRAGState,
    ):

        if not state.get(
            "is_complex",
            False,
        ):
            return "simple_retrieve"

        return [
            Send(
                "retrieve_subquestion",
                {
                    "index": index,
                    "subquestion": subquestion,
                },
            )
            for index, subquestion
            in enumerate(
                state.get(
                    "subquestions",
                    [],
                ),
                start=1,
            )
        ]

    def _simple_retrieve_node(
        self,
        state: ComplexRAGState,
    ) -> dict:

        print(
            "[ComplexRAG] "
            "Planner 判断为简单问题，"
            "进入原 Corrective RAG。"
        )

        return {
            "final_context": (
                self.simple_retrieve(
                    state[
                        "original_question"
                    ]
                )
            )
        }

    def _retrieve_subquestion_node(
        self,
        state: WorkerState,
    ) -> dict:

        index = state["index"]
        subquestion = state[
            "subquestion"
        ]

        print(
            "[ComplexRAG] "
            f"Worker {index} 开始检索："
            f"{subquestion}"
        )

        try:
            result = self.raw_retrieve(
                subquestion
            )

        except Exception as exc:
            result = (
                "该子问题检索失败："
                f"{exc}"
            )

        result = str(
            result
            or ""
        )[
            :self.max_each_result_chars
        ]

        print(
            "[ComplexRAG] "
            f"Worker {index} 检索完成"
        )

        return {
            "evidence_parts": [
                {
                    "index": index,
                    "subquestion": subquestion,
                    "evidence": result,
                }
            ]
        }

    def _merge_node(
        self,
        state: ComplexRAGState,
    ) -> dict:

        parts = list(
            state.get(
                "evidence_parts",
                [],
            )
        )

        parts.sort(
            key=lambda item: int(
                item.get(
                    "index",
                    0,
                )
            )
        )

        blocks = []

        for item in parts:
            blocks.append(
                (
                    f"【子问题 {item['index']}】\n"
                    f"{item['subquestion']}\n\n"
                    f"{item['evidence']}"
                )
            )

        merged = (
            "【Complex RAG：复杂问题拆解检索结果】\n\n"
            f"原始问题：{state['original_question']}\n\n"
            + "\n\n".join(blocks)
        )

        merged = merged[
            :self.max_merged_chars
        ]

        print(
            "[ComplexRAG] "
            f"已合并 {len(parts)} "
            "路子问题检索结果"
        )

        return {
            "merged_evidence": merged,
        }

    def _grade_merged_node(
        self,
        state: ComplexRAGState,
    ) -> dict:

        sufficient, reason = (
            self._grade_merged_evidence(
                state[
                    "original_question"
                ],
                state[
                    "merged_evidence"
                ],
            )
        )

        print(
            "[ComplexRAG] Coverage Grade："
            f"sufficient={sufficient} | "
            f"reason={reason}"
        )

        result = {
            "merged_sufficient": sufficient,
            "grade_reason": reason,
        }

        if sufficient:
            result[
                "final_context"
            ] = state[
                "merged_evidence"
            ]

        return result

    def _route_after_grade(
        self,
        state: ComplexRAGState,
    ) -> str:

        if state.get(
            "merged_sufficient",
            False,
        ):
            return "done"

        return "fallback"

    def _fallback_corrective_node(
        self,
        state: ComplexRAGState,
    ) -> dict:

        print(
            "[ComplexRAG] "
            "拆题后的证据仍不足，"
            "回退原 Corrective RAG。"
        )

        return {
            "final_context": (
                self.simple_retrieve(
                    state[
                        "original_question"
                    ]
                )
            )
        }

    def _build_graph(self):

        builder = StateGraph(
            ComplexRAGState
        )

        builder.add_node(
            "plan",
            self._plan_node,
        )
        builder.add_node(
            "simple_retrieve",
            self._simple_retrieve_node,
        )
        builder.add_node(
            "retrieve_subquestion",
            self._retrieve_subquestion_node,
        )
        builder.add_node(
            "merge",
            self._merge_node,
        )
        builder.add_node(
            "grade_merged",
            self._grade_merged_node,
        )
        builder.add_node(
            "fallback_corrective",
            self._fallback_corrective_node,
        )

        builder.add_edge(
            START,
            "plan",
        )

        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            [
                "simple_retrieve",
                "retrieve_subquestion",
            ],
        )

        builder.add_edge(
            "simple_retrieve",
            END,
        )

        builder.add_edge(
            "retrieve_subquestion",
            "merge",
        )

        builder.add_edge(
            "merge",
            "grade_merged",
        )

        builder.add_conditional_edges(
            "grade_merged",
            self._route_after_grade,
            {
                "done": END,
                "fallback": (
                    "fallback_corrective"
                ),
            },
        )

        builder.add_edge(
            "fallback_corrective",
            END,
        )

        return builder.compile()

    def run(
        self,
        question: str,
    ) -> str:

        question = str(
            question
            or ""
        ).strip()

        if not self.enabled:
            return self.simple_retrieve(
                question
            )

        if self._is_obviously_simple(
            question
        ):
            print(
                "[ComplexRAG] "
                "Fast Path：明显简单问题，"
                "跳过 Planner。"
            )

            return self.simple_retrieve(
                question
            )

        state = self.graph.invoke(
            {
                "original_question": (
                    question
                ),
                "evidence_parts": [],
            }
        )

        result = str(
            state.get(
                "final_context",
                "",
            )
            or ""
        ).strip()

        if result:
            return result

        print(
            "[ComplexRAG] "
            "Graph 未生成有效结果，"
            "回退普通 Corrective RAG。"
        )

        return self.simple_retrieve(
            question
        )
