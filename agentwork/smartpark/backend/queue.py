"""Non-blocking, sequential parking transaction queue."""

import queue
import threading
import time
from collections.abc import Callable


class TransactionQueue:
    def __init__(self, handler: Callable[[str], None], interval_seconds: int = 5):
        self.handler = handler
        self.interval_seconds = interval_seconds
        self.items: queue.Queue[str] = queue.Queue()
        self.pending: set[str] = set()
        self.lock = threading.Lock()
        self.running = True
        self.worker = threading.Thread(target=self._work, name="smartpark-transactions", daemon=True)
        self.worker.start()

    def add(self, plate: str) -> bool:
        normalized = plate.strip().upper()
        with self.lock:
            if not normalized or normalized in self.pending:
                return False
            self.pending.add(normalized)
            self.items.put(normalized)
            return True

    def _work(self) -> None:
        while self.running:
            try:
                plate = self.items.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.handler(plate)
            finally:
                with self.lock:
                    self.pending.discard(plate)
                self.items.task_done()
            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        self.running = False
