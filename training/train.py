from utils.techniques.gradient_clipping import clip_gradient
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from .checkpoint import CheckPoint, load
from logger import Logger


class Trainer(nn.Module):
    def __init__(self, model, train_loader, val_loader, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.metrics = model.metrics  # list
        self.set_attribute(kwargs)

    def logged(self, logs):
        tags = [tag for tag in logs.keys()]
        values = [value for value in logs.values()]

        self.logger.write(tags=tags, values=values)

    def fit(self, num_epochs=10, print_per_iter=None):
        self.num_epochs = num_epochs
        self.num_iters = num_epochs * len(self.train_loader)

        if self.checkpoint is None:
            self.checkpoint = CheckPoint(save_per_epoch=int(num_epochs/10) + 1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.train_loader) / 10)

        print('===========================START TRAINING=================================')
        for epoch in range(num_epochs):
            try:
                self.epoch = epoch + 1

                self.train_per_epoch()

                if self.epoch % self.evaluate_epoch == 0 and self.epoch + 1 >= self.evaluate_epoch:
                    self.evaluate_per_epoch()

                if self.scheduler is not None:
                    self.scheduler.step()

                if self.epoch % self.checkpoint.save_per_epoch == 0 or self.epoch == num_epochs:
                    self.checkpoint.save(self.model, epoch=self.epoch)
            except KeyboardInterrupt:
                self.checkpoint.save(self.model, epoch=self.epoch, interrupted=True)
                print('Stop training. Saved checkpoint!')
                break

            print('Train Completed')

    def train_per_epoch(self):
        self.model.train()
        running_time = 0
        running_loss = {}

        for i, batch in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            start_time = time.time()
            loss, loss_dict = self.model.training_step(batch)

            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()

            if self.gradient_clip is not None:
                clip_gradient(self.optimizer, self.gradient_clip)

            self.optimizer.step()
            end_time = time.time()

            for (key, value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time - start_time
            iters = len(self.train_loader)*self.epoch+i+1
            if iters % self.print_per_iter == 0:

                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter

                loss_string = '{}'.format(running_loss)[
                    1:-1].replace("'", '').replace(",", ' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f} s".format(
                    self.epoch, self.num_epochs, iters, self.num_iters, loss_string, running_time))
                self.logged(
                    {"Training Loss/Batch": running_loss['T'] / self.print_per_iter, })
                running_loss = {}
                running_time = 0

    def inference_per_batch(self, test_loader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model.inference_step(batch)
                if isinstance(outputs, (list, tuple)):
                    for i in outputs:
                        results.append(i)
                else:
                    results = outputs
                break

        return results

    def evaluate_per_epoch(self):
        self.model.eval()
        epoch_loss = {}
        metric_dict = {}

        print('===========================EVALUATION=================================')
        start_time = time.time()

        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                (loss, loss_dict), metrics = self.model.evaluate_step(batch)
                for key, value in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

                metric_dict.update(metrics)

        end_time = time.time()
        running_time = end_time - start_time
        self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.val_loader)

        loss_string = '{}'.format(epoch_loss)[
            1:-1].replace("'", '').replace(",", ' ||')

        print()
        print("[{}|{}] || {} || Acc: {:10.4f} || Time: {:10.4f} s".format(
            self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(f'{metric}:  + {score}', end=' | ')
        print('==')
        print('==========================================================================')

        log_dict = {"Validation Loss/Epoch": epoch_loss['T'] /
                    len(self.val_loader)}
        log_dict.update(metric_dict)
        self.logged(log_dict)

    def __str__(self) -> str:
        title = '------------- Model Summary ---------------\n'
        name = f'Name: {self.model.name}\n'
        params = f'Number of params: {self.model.trainable_parameters}\n'
        loss = f'Loss function: {self.criterion[:-2]} \n'
        train_iter_per_epoch = f'Number of train iterations per epoch: {len(self.train_loader)}\n'
        val_iter_per_epoch = f'Number of val iterations per epoch: {len(self.val_loader)}'

        return title + name + params + loss + train_iter_per_epoch + val_iter_per_epoch

    def print_forward_step(self):
        self.model.eval()
        outputs = self.model.forward_step()
        print('Feedforward: output_shape: ', outputs.shape)

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.evaluate_epoch = 1
        self.scheduler = None
        self.gradient_clip = None
        self.logger = None
        for i, j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()
