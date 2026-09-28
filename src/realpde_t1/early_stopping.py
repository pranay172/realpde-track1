"""Conservative training-only plateau detection for noisy minibatch losses."""
from dataclasses import dataclass
import math


@dataclass
class PlateauMonitor:
    window: int = 200
    min_updates: int = 1000
    patience: int = 600
    relative_improvement: float = 0.01
    best_mean: float = math.inf
    best_update: int = 0

    def observe(self, losses, update):
        if len(losses) < self.window:
            return {"ready":False,"stop":False,"improved":False}
        mean = sum(losses[-self.window:]) / self.window
        if not math.isfinite(mean):
            raise ValueError("nonfinite smoothed loss")
        improved = mean < self.best_mean * (1-self.relative_improvement)
        if improved:
            self.best_mean, self.best_update = mean, update
        stop = update >= self.min_updates and update-self.best_update >= self.patience
        return {"ready":True,"stop":stop,"improved":improved,"update":update,
                "mean_loss":mean,"best_mean_loss":self.best_mean,
                "best_update":self.best_update,"updates_without_improvement":update-self.best_update}
