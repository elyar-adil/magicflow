class Module:
    def __init__(self):
        self.device = None
        pass

    def parameters(self, recurse: bool = True):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def to(self, device=None):
        ...
