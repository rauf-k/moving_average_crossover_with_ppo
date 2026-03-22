from torch.utils.tensorboard import SummaryWriter


class TbLogger:
    def __init__(self):
        self.writer = SummaryWriter()
        self.tracker = {}

    def add_scalar(self, name, value):
        if name not in self.tracker.keys():
            self.tracker[name] = 0
        else:
            self.tracker[name] += 1
        self.writer.add_scalar(name, value, self.tracker[name])


WRITER = TbLogger()
