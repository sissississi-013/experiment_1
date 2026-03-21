from validation_pipeline.events import PipelineEvent
from validation_pipeline.persistence.run_store import RunStore


class PersistenceSubscriber:
    def __init__(self, store: RunStore, run_id: str):
        self.store = store
        self.run_id = run_id

    def __call__(self, event: PipelineEvent):
        try:
            self.store.store_event(self.run_id, event)
        except Exception:
            pass
