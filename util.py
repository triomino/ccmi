class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val, count=1):
        self.total += val * count
        self.steps += count

    def value(self):
        return self.total/float(self.steps) if self.steps > 0 else 0
