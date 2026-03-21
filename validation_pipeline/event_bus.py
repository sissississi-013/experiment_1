import threading
from typing import Callable
from validation_pipeline.events import PipelineEvent

class EventBus:
    def __init__(self):
        self._subscribers: dict[type, list[Callable]] = {}
        self._all_subscribers: list[Callable] = []
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[PipelineEvent], callback: Callable):
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: Callable):
        with self._lock:
            self._all_subscribers.append(callback)

    def publish(self, event: PipelineEvent):
        with self._lock:
            callbacks = list(self._subscribers.get(type(event), []))
            all_callbacks = list(self._all_subscribers)
        for cb in callbacks + all_callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def clear(self):
        with self._lock:
            self._subscribers.clear()
            self._all_subscribers.clear()
