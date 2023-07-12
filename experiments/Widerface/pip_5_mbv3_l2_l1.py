
class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 256
        self.init_lr = 0.001
        self.num_epochs = 201
        self.decay_steps = [30, 50]
        self.input_size = 128
        self.backbone = 'mobilenet_v3_small'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 2
        self.reg_loss_weight = 1
        self.num_lms = 5
        self.save_interval = self.num_epochs
        self.use_gpu = True
        self.gpu_id = 0
