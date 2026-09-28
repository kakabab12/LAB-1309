"""모델 없이 시뮬레이터만 띄운다 — 스크립트 전문가 개발·시험용 (GPU 거의 안 씀)."""
import switch_experiment as sx
from libero.libero import benchmark


class SimRunner:
    def __init__(self, task_a=0, task_b=None):
        a = sx.fill_defaults(sx.default_args())
        a.task_a, a.task_b, a.strategy = task_a, task_b, "none"
        self.args = a
        self.suite = benchmark.get_benchmark_dict()[sx.SUITE]()
        self.chk_a = sx.GoalChecker(self.suite, task_a)
        self.chk_b = sx.GoalChecker(self.suite, task_b) if task_b is not None else None
        self.policy = None


def episode(task, i, task_b=None):
    r = SimRunner(task, task_b)
    return r, sx.Episode(r, i)
