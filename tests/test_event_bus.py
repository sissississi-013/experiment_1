from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import PipelineEvent, ModuleStarted, ModuleCompleted, ImageProgress

def test_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe(ModuleStarted, lambda e: received.append(e))
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 1
    assert received[0].module == "test"

def test_subscribe_does_not_receive_other_types():
    bus = EventBus()
    received = []
    bus.subscribe(ModuleStarted, lambda e: received.append(e))
    bus.publish(ModuleCompleted(module="test"))
    assert len(received) == 0

def test_subscribe_all_receives_everything():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda e: received.append(e))
    bus.publish(ModuleStarted(module="a"))
    bus.publish(ModuleCompleted(module="b"))
    bus.publish(ImageProgress(module="c", current=1, total=10, image_path="/x.jpg"))
    assert len(received) == 3

def test_subscriber_error_does_not_kill_pipeline():
    bus = EventBus()
    received = []
    def bad_subscriber(event):
        raise RuntimeError("subscriber crash")
    def good_subscriber(event):
        received.append(event)
    bus.subscribe_all(bad_subscriber)
    bus.subscribe_all(good_subscriber)
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 1

def test_clear_removes_all_subscribers():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda e: received.append(e))
    bus.clear()
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 0

def test_multiple_subscribers_same_type():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(ModuleStarted, lambda e: a.append(e))
    bus.subscribe(ModuleStarted, lambda e: b.append(e))
    bus.publish(ModuleStarted(module="test"))
    assert len(a) == 1
    assert len(b) == 1
