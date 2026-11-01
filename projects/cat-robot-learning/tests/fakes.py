"""Fake robot HTTP API for acquisition tests."""


def sample(n, time_ms, armed=False, epoch=0, left=0, right=0, reason="startup", invalid=0):
    return {
        "sample": n,
        "time_ms": time_ms,
        "epoch": epoch,
        "armed": armed,
        "reason": reason,
        "encoder_left": left,
        "encoder_right": right,
        "encoder_left_invalid": invalid,
        "encoder_right_invalid": 0,
        "sync_led": False,
    }


class FakeRobot:
    """Device clock advances 20 ms per sample; rotation passes 1.3 m after arming."""

    def __init__(self, drop_at=None, fail_armed_reads=0, student_id=None):
        self.fail_armed_reads = fail_armed_reads
        self.student_id = student_id
        self.drives = []
        self.armed = False
        self.samples = []
        self.drop_at = drop_at
        self.posts = []
        self.epoch = 0
        self.armed_at = 0

    def _advance(self, count=4):
        for _ in range(count):
            n = len(self.samples) + 1
            moved = (n - self.armed_at) * 300 if self.armed else 0
            self.samples.append(
                sample(n, 5000 + 20 * n, self.armed, self.epoch, moved, -moved, "operator")
            )

    def get(self, path, **params):
        if path == "/api/status":
            return {
                "armed": self.armed,
                "encoder_left": 0,
                "encoder_right": 0,
                "firmware": "fixture",
                "boot_id": 9,
                "student_id": self.student_id or "unfitted",
                "student_measured": self.student_id is not None,
            }
        if self.armed and self.fail_armed_reads:
            self.fail_armed_reads -= 1
            import httpx

            raise httpx.ReadTimeout("fixture telemetry failure")
        self._advance()
        after = params.get("after", len(self.samples))
        batch = [s for s in self.samples if s["sample"] > after][:16]
        dropped = 0
        if self.drop_at and batch and batch[0]["sample"] <= self.drop_at <= batch[-1]["sample"]:
            batch = [s for s in batch if s["sample"] != self.drop_at]
            dropped = 1
        return {"latest": len(self.samples), "boot_id": 9, "dropped": dropped, "samples": batch}

    def post(self, path, **fields):
        self.posts.append(path)
        if path == "/api/drive":
            self.drives.append(fields)
        if path == "/api/arm":
            self.armed, self.epoch, self.armed_at = True, 1, len(self.samples)
            return {"epoch": 1}
        if path == "/api/stop":
            self.armed = False
        return {"ok": True}

    def calibrate(self, config):
        pass

    def close(self):
        pass
