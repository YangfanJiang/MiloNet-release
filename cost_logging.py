import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str, model: str = "") -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 4))


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_usage(response: Any) -> Dict[str, Optional[int]]:
    candidates = [
        _get_attr_or_key(response, "raw"),
        _get_attr_or_key(response, "additional_kwargs"),
        response,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        usage = _get_attr_or_key(candidate, "usage")
        if not usage and isinstance(candidate, dict):
            usage = candidate.get("token_usage")
        if not usage:
            continue
        prompt = _get_attr_or_key(usage, "prompt_tokens")
        completion = _get_attr_or_key(usage, "completion_tokens")
        total = _get_attr_or_key(usage, "total_tokens")
        if prompt is None:
            prompt = _get_attr_or_key(usage, "input_tokens")
        if completion is None:
            completion = _get_attr_or_key(usage, "output_tokens")
        try:
            return {
                "prompt_tokens": int(prompt) if prompt is not None else None,
                "completion_tokens": int(completion) if completion is not None else None,
                "total_tokens": int(total) if total is not None else None,
            }
        except (TypeError, ValueError):
            continue
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def response_text(response: Any) -> str:
    text = _get_attr_or_key(response, "text")
    if text is not None:
        return str(text)
    message = _get_attr_or_key(response, "message")
    if message is not None:
        content = _get_attr_or_key(message, "content")
        if content is not None:
            return str(content)
    return str(response)


class CostLogger:
    def __init__(self, path: Optional[str], system: str, model: str = ""):
        self.path = Path(path) if path else None
        self.system = system
        self.model = model
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def log(
        self,
        *,
        stage: str,
        case_id: Optional[str],
        model: str,
        prompt: str,
        completion: str,
        latency_seconds: float,
        success: bool = True,
        error: Optional[str] = None,
        response: Any = None,
    ) -> None:
        if not self.path:
            return
        usage = extract_usage(response)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        estimated = False
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt, model)
            estimated = True
        if completion_tokens is None:
            completion_tokens = estimate_tokens(completion, model)
            estimated = True
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        row = {
            "system": self.system,
            "case_id": case_id,
            "stage": stage,
            "model": model or self.model,
            "success": success,
            "latency_seconds": latency_seconds,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_estimated": estimated,
        }
        if error:
            row["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_counts(
        self,
        *,
        stage: str,
        case_id: Optional[str],
        model: str,
        latency_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
        tokens_estimated: bool = False,
    ) -> None:
        if not self.path:
            return
        row = {
            "system": self.system,
            "case_id": case_id,
            "stage": stage,
            "model": model or self.model,
            "success": success,
            "latency_seconds": latency_seconds,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens if total_tokens is not None else (prompt_tokens or 0) + (completion_tokens or 0)),
            "tokens_estimated": tokens_estimated,
        }
        if error:
            row["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def complete(self, llm: Any, prompt: str, *, stage: str, case_id: Optional[str] = None) -> Any:
        model = str(getattr(llm, "model", None) or getattr(llm, "model_name", None) or self.model)
        start = time.perf_counter()
        try:
            response = llm.complete(prompt=prompt)
        except Exception as exc:
            self.log(
                stage=stage,
                case_id=case_id,
                model=model,
                prompt=prompt,
                completion="",
                latency_seconds=time.perf_counter() - start,
                success=False,
                error=str(exc),
            )
            raise
        self.log(
            stage=stage,
            case_id=case_id,
            model=model,
            prompt=prompt,
            completion=response_text(response),
            latency_seconds=time.perf_counter() - start,
            response=response,
        )
        return response


_PATCH_INSTALLED = False


class LlamaIndexCostLoggingHandler:
    def __init__(self, path: str, system: str):
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
        from llama_index.core.callbacks import CBEventType

        class _Handler(BaseCallbackHandler):
            def __init__(self, outer: "LlamaIndexCostLoggingHandler"):
                super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
                self.outer = outer

            def on_event_start(
                self,
                event_type: Any,
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                parent_id: str = "",
                **kwargs: Any,
            ) -> str:
                return self.outer.on_event_start(event_type, payload, event_id, parent_id, **kwargs)

            def on_event_end(
                self,
                event_type: Any,
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                **kwargs: Any,
            ) -> None:
                self.outer.on_event_end(event_type, payload, event_id, **kwargs)

            def start_trace(self, trace_id: Optional[str] = None) -> None:
                return

            def end_trace(
                self,
                trace_id: Optional[str] = None,
                trace_map: Optional[Dict[str, List[str]]] = None,
            ) -> None:
                return

        self.logger = CostLogger(path, system)
        self.starts: Dict[str, Dict[str, Any]] = {}
        self.handler = _Handler(self)
        self.llm_event_type = CBEventType.LLM

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type != self.llm_event_type:
            return event_id
        payload = payload or {}
        self.starts[event_id] = {
            "time": time.perf_counter(),
            "model": self._extract_model(payload),
        }
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != self.llm_event_type:
            return

        from llama_index.core.callbacks import EventPayload
        from llama_index.core.callbacks.token_counting import TokenCounter, get_llm_token_counts, get_tokenizer

        payload = payload or {}
        started = self.starts.pop(event_id, {})
        latency_seconds = time.perf_counter() - float(started.get("time", time.perf_counter()))
        model = self._extract_model(payload) or str(started.get("model") or "")
        case_id = os.getenv("MILONET_COST_LOG_CASE_ID")
        stage = os.getenv("MILONET_COST_LOG_STAGE", "llama_index_callback")

        exception = payload.get(EventPayload.EXCEPTION)
        if exception is not None:
            self.logger.log_counts(
                stage=stage,
                case_id=case_id,
                model=model,
                latency_seconds=latency_seconds,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(exception),
                tokens_estimated=True,
            )
            return

        token_counter = TokenCounter(tokenizer=get_tokenizer())
        counts = get_llm_token_counts(token_counter=token_counter, payload=payload, event_id=event_id)
        response = payload.get(EventPayload.COMPLETION) or payload.get(EventPayload.RESPONSE)
        usage = extract_usage(response)
        tokens_estimated = not bool(usage.get("prompt_tokens") or usage.get("completion_tokens") or usage.get("total_tokens"))
        self.logger.log_counts(
            stage=stage,
            case_id=case_id,
            model=model,
            latency_seconds=latency_seconds,
            prompt_tokens=counts.prompt_token_count,
            completion_tokens=counts.completion_token_count,
            total_tokens=counts.total_token_count,
            tokens_estimated=tokens_estimated,
        )

    @staticmethod
    def _extract_model(payload: Dict[str, Any]) -> str:
        try:
            from llama_index.core.callbacks import EventPayload

            serialized = payload.get(EventPayload.SERIALIZED) or {}
            if isinstance(serialized, dict):
                return str(serialized.get("model") or serialized.get("model_name") or "")
        except Exception:
            pass
        return ""


def install_llama_index_cost_logging_from_env() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    path = os.getenv("MILONET_COST_LOG_PATH")
    if not path:
        return
    system = os.getenv("MILONET_COST_LOG_SYSTEM", "MiloNet")
    try:
        from llama_index.core import Settings
        from llama_index.core.callbacks import CallbackManager
    except Exception:
        return

    cost_handler = LlamaIndexCostLoggingHandler(path, system).handler
    existing_manager = getattr(Settings, "_callback_manager", None)
    handlers = list(getattr(existing_manager, "handlers", []) or [])
    if not any(handler.__class__ is cost_handler.__class__ for handler in handlers):
        handlers.append(cost_handler)
    Settings.callback_manager = CallbackManager(handlers)
    _PATCH_INSTALLED = True
