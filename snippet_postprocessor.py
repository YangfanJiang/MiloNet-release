import re
from typing import Iterable, List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode


class RelevantSnippetPostprocessor(BaseNodePostprocessor):
    """Trim retrieved nodes to the most relevant sentences."""

    def __init__(self, max_chars: int = 1200, sentence_window: int = 1) -> None:
        self._max_chars = max_chars
        self._sentence_window = sentence_window

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        query = (query_bundle.query_str or "") if query_bundle else ""
        keywords = self._extract_keywords(query)
        processed: List[NodeWithScore] = []

        for node_with_score in nodes:
            base_node = node_with_score.node
            if not isinstance(base_node, TextNode):
                processed.append(node_with_score)
                continue

            original_text = base_node.get_content(MetadataMode.NONE)
            snippet = self._build_snippet(original_text, keywords)
            updated_text = snippet if snippet else original_text[: self._max_chars]
            node_copy = base_node.model_copy(update={"text": updated_text})
            processed.append(
                NodeWithScore(node=node_copy, score=node_with_score.score)
            )

        return processed

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"\w+", query.lower())
        return [token for token in tokens if len(token) > 2]

    def _build_snippet(self, text: str, keywords: Iterable[str]) -> str:
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        matches: List[str] = []

        if keywords:
            lowered_keywords = list(keywords)
            for idx, sentence in enumerate(sentences):
                lowered_sentence = sentence.lower()
                if any(keyword in lowered_sentence for keyword in lowered_keywords):
                    start = max(idx - self._sentence_window, 0)
                    end = min(idx + self._sentence_window + 1, len(sentences))
                    matches.extend(sentences[start:end])
        else:
            matches = sentences[: 2 * self._sentence_window + 1]

        if not matches:
            matches = sentences[:1]

        snippet = " ".join(self._deduplicate(matches)).strip()
        if len(snippet) > self._max_chars:
            snippet = snippet[: self._max_chars].rstrip() + "..."
        return snippet

    def _deduplicate(self, sentences: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for sentence in sentences:
            if sentence and sentence not in seen:
                seen.add(sentence)
                ordered.append(sentence)
        return ordered
