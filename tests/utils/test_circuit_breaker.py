import unittest

from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def boom():
    raise ConnectionError("down")


class CircuitBreakerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=10.0, trip_on=(ConnectionError,), clock=self.clock)

    def test_opens_after_consecutive_failures_and_skips_calls(self):
        calls = []

        def fn():
            calls.append(1)
            raise ConnectionError("down")

        for _ in range(3):
            with self.assertRaises(ConnectionError):
                self.cb.call(fn)
        self.assertEqual(self.cb.state, "open")
        with self.assertRaises(CircuitOpenError):
            self.cb.call(fn)
        self.assertEqual(len(calls), 3)  # 開いた後は依存先に触れない

    def test_success_resets_failure_count(self):
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.assertEqual(self.cb.call(lambda: "ok"), "ok")
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.assertEqual(self.cb.state, "closed")

    def test_fallback_used_while_open(self):
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.assertEqual(self.cb.call(boom, fallback=lambda: "cached"), "cached")

    def test_half_open_allows_single_probe_then_closes_on_success(self):
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.clock.t = 10.0
        self.assertEqual(self.cb.state, "half_open")
        self.assertTrue(self.cb.allow())
        self.assertFalse(self.cb.allow())  # 試行中はほかを通さない
        self.cb.record_success()
        self.assertEqual(self.cb.state, "closed")
        self.assertTrue(self.cb.allow())

    def test_half_open_failure_reopens_for_full_timeout(self):
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.clock.t = 10.0
        with self.assertRaises(ConnectionError):
            self.cb.call(boom)
        self.assertEqual(self.cb.state, "open")
        self.clock.t = 19.9
        self.assertEqual(self.cb.state, "open")
        self.clock.t = 20.0
        self.assertEqual(self.cb.state, "half_open")

    def test_untracked_exception_does_not_trip(self):
        def bug():
            raise ValueError("not a connectivity problem")

        for _ in range(10):
            with self.assertRaises(ValueError):
                self.cb.call(bug)
        self.assertEqual(self.cb.state, "closed")

    def test_untracked_exception_in_half_open_releases_probe_slot(self):
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                self.cb.call(boom)
        self.clock.t = 10.0
        with self.assertRaises(ValueError):
            self.cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        self.assertTrue(self.cb.allow())


if __name__ == "__main__":
    unittest.main()
