import threading
import time

from smartpark.backend.queue import TransactionQueue


def test_queue_deduplicates_and_processes():
    processed = []
    ready = threading.Event()

    def handler(plate):
        processed.append(plate)
        ready.set()

    queue = TransactionQueue(handler, interval_seconds=0)
    assert queue.add("abc")
    assert not queue.add("ABC")
    assert ready.wait(2)
    time.sleep(0.05)
    queue.stop()
    assert processed == ["ABC"]
