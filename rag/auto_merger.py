from __future__ import annotations

from app.core.stream_events import event_print as print
import json
from pathlib import Path

from langchain_core.documents import (
    Document,
)

from .retriever import (
    RetrievalResult,
)


class AutoMerger:
    """
    Parent-Child Auto-Merging。

    如果同一个 Parent 下，
    有足够多的 Child 同时被检索到，
    就把多个 Child 替换成完整 Parent。
    """

    def __init__(
        self,
        index_path: str,
        settings: dict | None = None,
    ):
        settings = (
            settings
            or {}
        )

        self.enabled = bool(
            settings.get(
                "enabled",
                True,
            )
        )

        self.min_child_hits = max(
            2,
            int(
                settings.get(
                    "min_child_hits",
                    2,
                )
            ),
        )

        self.merge_ratio = float(
            settings.get(
                "merge_ratio",
                0.5,
            )
        )

        self.merge_ratio = max(
            0.0,
            min(
                1.0,
                self.merge_ratio,
            ),
        )

        self.max_parent_chars = max(
            500,
            int(
                settings.get(
                    "max_parent_chars",
                    2400,
                )
            ),
        )

        self.parent_store_path = (
            Path(index_path)
            / "parent_store.json"
        )

        self.parents = {}

        if not self.enabled:
            print(
                "ℹ️ AutoMerge 已关闭"
            )
            return

        if not (
            self.parent_store_path
            .exists()
        ):
            print(
                "⚠️ 没有找到 "
                "parent_store.json，"
                "AutoMerge 自动降级关闭"
            )

            self.enabled = False
            return

        try:
            with (
                self.parent_store_path
                .open(
                    "r",
                    encoding="utf-8",
                )
            ) as file:
                payload = json.load(
                    file
                )

            self.parents = dict(
                payload.get(
                    "parents",
                    {},
                )
            )

        except Exception as exc:
            print(
                "⚠️ Parent Store "
                "加载失败，"
                "AutoMerge 自动关闭："
                f"{exc}"
            )

            self.enabled = False
            self.parents = {}

            return

        print(
            "✅ AutoMerger 已加载："
            f"{len(self.parents)} "
            "个 Parent Chunk"
        )

    @staticmethod
    def _minimum(
        values,
    ):
        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return None

        return min(
            values
        )

    @staticmethod
    def _maximum(
        values,
    ):
        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return None

        return max(
            values
        )

    def _create_parent_result(
        self,
        parent_id: str,
        child_results: list,
        total_children: int,
        hit_ratio: float,
    ) -> RetrievalResult:

        parent_record = (
            self.parents[
                parent_id
            ]
        )

        content = str(
            parent_record.get(
                "page_content",
                "",
            )
            or ""
        )

        truncated = False

        if (
            len(content)
            > self.max_parent_chars
        ):
            content = content[
                :self.max_parent_chars
            ]

            truncated = True

        metadata = dict(
            parent_record.get(
                "metadata",
                {},
            )
            or {}
        )

        metadata.update(
            {
                "chunk_level":
                    "parent",

                "auto_merged":
                    True,

                "matched_child_count":
                    len(
                        child_results
                    ),

                "parent_child_count":
                    total_children,

                "matched_child_ratio":
                    round(
                        hit_ratio,
                        4,
                    ),

                "parent_truncated":
                    truncated,
            }
        )

        document = Document(
            page_content=content,
            metadata=metadata,
        )

        fusion_score = max(
            float(
                item.fusion_score
            )
            for item
            in child_results
        )

        return RetrievalResult(
            document=document,

            fusion_score=(
                fusion_score
            ),

            vector_rank=(
                self._minimum(
                    [
                        item.vector_rank
                        for item
                        in child_results
                    ]
                )
            ),

            bm25_rank=(
                self._minimum(
                    [
                        item.bm25_rank
                        for item
                        in child_results
                    ]
                )
            ),

            vector_distance=(
                self._minimum(
                    [
                        item.vector_distance
                        for item
                        in child_results
                    ]
                )
            ),

            bm25_score=(
                self._maximum(
                    [
                        item.bm25_score
                        for item
                        in child_results
                    ]
                )
            ),
        )

    def merge(
        self,
        candidates: list[
            RetrievalResult
        ],
    ) -> list[
        RetrievalResult
    ]:

        if (
            not self.enabled
            or not candidates
        ):
            return candidates

        groups = {}

        passthrough = []

        for item in candidates:

            document = (
                item.document
            )

            metadata = (
                document.metadata
                or {}
            )

            parent_id = (
                metadata.get(
                    "parent_id"
                )
            )

            chunk_level = (
                metadata.get(
                    "chunk_level"
                )
            )

            if (
                chunk_level == "child"
                and parent_id
            ):
                groups.setdefault(
                    str(parent_id),
                    [],
                ).append(
                    item
                )

            else:
                passthrough.append(
                    item
                )

        output = list(
            passthrough
        )

        merged_parent_count = 0

        for (
            parent_id,
            child_results,
        ) in groups.items():

            first_metadata = (
                child_results[0]
                .document
                .metadata
                or {}
            )

            try:
                total_children = max(
                    1,
                    int(
                        first_metadata.get(
                            "parent_child_count",
                            len(
                                child_results
                            ),
                        )
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                total_children = max(
                    1,
                    len(
                        child_results
                    ),
                )

            hit_count = len(
                child_results
            )

            hit_ratio = min(
                1.0,
                (
                    hit_count
                    / total_children
                ),
            )

            should_merge = (
                hit_count
                >= self.min_child_hits
                and hit_ratio
                >= self.merge_ratio
                and parent_id
                in self.parents
            )

            if not should_merge:
                output.extend(
                    child_results
                )
                continue

            parent_result = (
                self._create_parent_result(
                    parent_id=parent_id,
                    child_results=(
                        child_results
                    ),
                    total_children=(
                        total_children
                    ),
                    hit_ratio=(
                        hit_ratio
                    ),
                )
            )

            output.append(
                parent_result
            )

            merged_parent_count += 1

            print(
                "[AutoMerge] "
                "Parent 提升："
                f"{parent_id} | "
                f"命中 "
                f"{hit_count}/"
                f"{total_children} "
                f"Child | "
                f"ratio="
                f"{hit_ratio:.2f}"
            )

        output.sort(
            key=lambda item:
                float(
                    item.fusion_score
                ),
            reverse=True,
        )

        if merged_parent_count:
            print(
                "[AutoMerge] "
                f"本轮共提升 "
                f"{merged_parent_count} "
                "个 Parent"
            )

        return output
