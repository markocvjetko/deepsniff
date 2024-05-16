import torchvision
import pytorch_lightning as pl
torchvision.disable_beta_transforms_warning()
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score 
import pytorch_lightning as pl
torchvision.disable_beta_transforms_warning()

class DeepSniff(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        #self.save_hyperparameters(ignore=['model', 'loss_function', 'optimizer', 'scheduler'])
        #self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return [self.optimizer], {'scheduler': self.scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        #self.log('pred', pred[0], sync_dist=True)
        #self.log('target', y[0], sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        #self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)
        #TODO add schedular step
         
        #if self.trainer.is_last_batch:
        # if self.lr_schedulers() is not None:
        #     if self.trainer.is_last_batch:
        #         self.lr_schedulers().step()
        #         print('lr: ' + str(self.trainer.optimizers[0].param_groups[0]['lr']))
                
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log('test_loss', loss, sync_dist=True)
        return loss


class DeepSniffClassification(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler, ignore_index=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.accuracy_train = MulticlassAccuracy(num_classes=model.output_dim, ignore_index=ignore_index)
        self.accuracy_val = self.accuracy_train.clone()
        self.accuracy_test = self.accuracy_train.clone()
        self.f1_train = MulticlassF1Score(num_classes=model.output_dim, ignore_index=ignore_index)
        self.f1_val = self.f1_train.clone()
        self.f1_test = self.f1_train.clone()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return [self.optimizer], {'scheduler': self.scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss, sync_dist=True)

        self.accuracy_train.update(pred, y)
        self.f1_train.update(pred, y)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy_val.update(pred, y)
        self.f1_val.update(pred, y)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy_test.update(pred, y)
        self.f1_test.update(pred, y)

        return loss

    def on_train_epoch_start (self):
         self.accuracy_train.reset()
         self.f1_train.reset()

    def on_validation_epoch_start(self):
        self.accuracy_val.reset()
        self.f1_val.reset()

    def on_test_epoch_start(self):
        self.accuracy_test.reset()
        self.f1_test.reset()

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     self.accuracy_train(outputs, batch[1])
    #     self.f1_train(outputs, batch[1])

    # def on_validation_batch_end(self, outputs, batch, batch_idx):
    #     print(outputs.shape, batch[1].shape)
    #     self.accuracy_val(outputs, batch[1])
    #     self.f1_val(outputs, batch[1])

    # def on_test_batch_end(self, outputs, batch, batch_idx):
    #     self.accuracy_test(outputs, batch[1])
    #     self.f1_test(outputs, batch[1])
    
    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy_train.compute(), sync_dist=True)
        self.log('train_f1_epoch', self.f1_train.compute(), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy_val.compute(), sync_dist=True)
        self.log('val_f1_epoch', self.f1_val.compute(), sync_dist=True)

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy_test.compute(), sync_dist=True)
        self.log('test_f1_epoch', self.f1_test.compute(), sync_dist=True)

    