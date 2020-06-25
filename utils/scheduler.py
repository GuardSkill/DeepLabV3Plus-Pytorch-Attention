from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # print('min_lr'+str(type(self.min_lr)))
        # for base_lr in self.base_lrs:
            # print('base_lr' + str(type((base_lr * (1 - self.last_epoch / self.max_iters) ** self.power))))
            # print('base_lr' + str(type((base_lr))))
            # print('self.last_epoch' + str(type((self.last_epoch))))
            # print('self.max_iters' + str(type((self.max_iters))))
            # print('self.power' + str(type((self.power))))
            # print('base_lr' + str((base_lr)))
            # print('self.last_epoch' + str((self.last_epoch)))
            # print('self.max_iters' + str((self.max_iters)))
            # print('self.power' + str((self.power)))
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]