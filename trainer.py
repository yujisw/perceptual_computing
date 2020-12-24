import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import numpy as np

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            loss_name = self.loss.__name__
            self.loss = torch.nn.DataParallel(self.loss).to(self.device)
            self.loss.__name__ = loss_name
            for metric in self.metrics:
                metric = torch.nn.DataParallel(metric).to(self.device)
        else:
            self.model.to(self.device)
            self.loss.to(self.device)
            for metric in self.metrics:
                metric.to(self.device)
        # self.model.to(self.device)
        # self.loss.to(self.device)
        # for metric in self.metrics:
        #     metric.to(self.device)


    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                # if self.device =='cuda':
                #     x, y = torch.nn.DataParallel(x), torch.nn.DataParallel(y)
                # else:
                #     x, y = x.to(self.device), y.to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        if self.device =='cuda':
            loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            if self.device =='cuda':
                loss = loss.sum()
        return loss, prediction
    

class TTAEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='TTA',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()
        
    def flip(self, x, axis):
        pred = self.model.forward(x.flip(axis).to(self.device))
        return pred.flip(axis)
    
    def add_noise(self, x, NUM=10000):
        xx = x.cpu().numpy().copy()
        b, c, h, w = xx.shape
        pts_x = np.random.randint(0, w-1 , NUM)
        pts_y = np.random.randint(0, h-1 , NUM)
        pts_b = np.random.randint(0, b-1 , NUM)
        xx = xx.transpose(0, 2, 3, 1)
        xx[(pts_b, pts_y,pts_x)] = (255, 255, 255)
        xx_ = torch.tensor(xx.transpose(0, 3, 1, 2))
        pred = self.model.forward(xx_.to(self.device))
        return pred

    def batch_update(self, x, y):
        with torch.no_grad():
            pred1 = self.model.forward(x)
            pred2 = self.flip(x, 2)
            pred3 = self.flip(x, 3)
            pred4 = self.add_noise(x)
#             prediction = (pred1+pred2+pred3)/3.0
            prediction = (pred1+pred2+pred3+pred4)/4.0
            loss = self.loss(prediction, y)
            if self.device =='cuda':
                loss = loss.sum()
        return loss, prediction


class TTA():
    def __init__(self, model, device='cpu'):
        self.model=model
        self.device=device        
        self.model.eval()
        
    def flip(self, x, axis):
        pred = self.model.forward(x.flip(axis).to(self.device))
        return pred.flip(axis)
    
    def add_noise(self, x, NUM=10000):
        xx = x.cpu().numpy().copy()
        b, c, h, w = xx.shape
        pts_x = np.random.randint(0, w-1 , NUM)
        pts_y = np.random.randint(0, h-1 , NUM)
        pts_b = np.random.randint(0, b-1 , NUM)
        xx = xx.transpose(0, 2, 3, 1)
        xx[(pts_b, pts_y,pts_x)] = (255, 255, 255)
        xx_ = torch.tensor(xx.transpose(0, 3, 1, 2))
        pred = self.model.forward(xx_.to(self.device))
        return pred

    def batch_update(self, x):
        with torch.no_grad():
            pred1 = self.model.forward(x)
            pred2 = self.flip(x, 2)
            pred3 = self.flip(x, 3)
#             pred4 = self.add_noise(x)
            prediction = (pred1+pred2+pred3)/3.0

        return prediction
