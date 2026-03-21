from unittest.mock import MagicMock
from validation_pipeline.persistence.subscriber import PersistenceSubscriber
from validation_pipeline.events import ModuleStarted, ImageVerdict


def test_subscriber_calls_store_event():
    store = MagicMock()
    sub = PersistenceSubscriber(store, "run123")
    event = ModuleStarted(module="test")
    sub(event)
    store.store_event.assert_called_once_with("run123", event)


def test_subscriber_swallows_errors():
    store = MagicMock()
    store.store_event.side_effect = Exception("DB down")
    sub = PersistenceSubscriber(store, "run123")
    sub(ModuleStarted(module="test"))  # Should not raise


def test_subscriber_passes_run_id():
    store = MagicMock()
    sub = PersistenceSubscriber(store, "my-run-id")
    sub(ImageVerdict(module="executor", image_id="x", image_path="/x.jpg", verdict="usable"))
    assert store.store_event.call_args[0][0] == "my-run-id"
