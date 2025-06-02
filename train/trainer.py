import torch
import os
import numpy as np
from utils import running_print, targets_print, outputs_print, error_print, warning_print
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, criterion, lr_scheduler=None):
        self.train_dataloader: DataLoader = train_dataloader
        self.test_dataloader: DataLoader = test_dataloader
        self.optimizer: Optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Module = model.to(self.device)

        self.loss_per_batch = []
        self.loss_per_epoch = []
        self.train_check_num = 0
        self.train_correct_num = 0
        self.train_acc = [0.]
        self.accuracies = [0.]
        self.num_trains = 0
        self.num_evals = 0

        self.save_print_loss_step = 10

        current_time = datetime.now()
        formatted_time = current_time.strftime("%d-%H-%M")
        logs_path = f'../logs/tensorboard_logs/tb_logs_{formatted_time}/'
        os.makedirs(logs_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logs_path)
        pass

    def train_step(self, inputs, labels, batch_idx, gradient_accumulation_steps, scheduler,
                   save_print_loss_step=10, print_targets_outputs_step=100):
        if isinstance(inputs, tuple):
            inputs = (item.to(self.device) for item in inputs)
        else:
            inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # running_print('Propagating forward...')
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss_item = loss.item()

        if gradient_accumulation_steps:
            loss = loss / gradient_accumulation_steps

        # running_print('Propagating backward...')
        loss.backward()

        if gradient_accumulation_steps:
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()
                # running_print('Updated weight.')

        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()
            # running_print('Updated weight.')

        if save_print_loss_step:
            if (batch_idx + 1) % save_print_loss_step == 0:
                self.loss_per_batch.append(loss_item)
                running_print(f'Batch {batch_idx + 1} end, num trains: {self.num_trains + 1},',
                              'training loss: {:.5f}'.format(loss_item))

        if print_targets_outputs_step:
            if (batch_idx + 1) % print_targets_outputs_step == 0:
                max_values, max_indices = torch.max(outputs, dim=1)
                targets_print(labels)
                outputs_print(max_indices)
                self.train_correct_num += (labels == max_indices).sum().item()
                self.train_check_num += max_indices.size(0)

        self.num_trains += 1
        self.writer.add_scalar('Loss/batch', loss_item, self.num_trains)
        pass

    def eval(self):
        with torch.no_grad():
            total = 0
            correct = 0
            for batch_idx, (inputs, labels) in enumerate(self.test_dataloader):
                # 将输入数据和标签转移到GPU上
                if isinstance(inputs, tuple):
                    inputs = (item.to(self.device) for item in inputs)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # 前向传播
                outputs = self.model(inputs)
                # 预测结果
                _, predicted = torch.max(outputs.data, dim=1)
                targets_print(labels)
                max_values, max_indices = torch.max(outputs, dim=1)
                outputs_print(max_indices)
                # 统计正确预测的样本数和总样本数
                total += labels.size(0)
                _correct = (predicted == labels).sum().item()
                correct += _correct
                running_print(f'Num test: batch {batch_idx + 1} end, '
                              f'batch correct quantity: {_correct}')

            # 输出精度
            accuracy = 100 * correct / (total + 1e-5)
            self.accuracies.append(accuracy)
            self.num_evals += 1
            self.writer.add_scalar('AccPerEval', accuracy, self.num_evals)
            running_print('Test accuracy: {:.2f}%'.format(accuracy))
        pass

    def train(self, num_epochs, num_epochs_per_test, num_epochs_per_plot, num_epochs_per_save,
              model_saving_path, lr_scheduler_kwargs=None, gradient_accumulation_steps=None):
        scheduler = None
        num_one_epoch_loss_item = None
        if lr_scheduler_kwargs:
            if self.lr_scheduler is None:
                raise ValueError('You did not set up a LR scheduler when then trainer initialized, '
                                 'but give the lr scheduler kwargs.')
            scheduler = self.lr_scheduler(self.optimizer, **lr_scheduler_kwargs)

        running_print('Train begin.')
        for epoch in range(num_epochs):
            # try:
            running_print(f'Epoch {epoch + 1} begin, LR: {self.optimizer.param_groups[0]["lr"]:.6f}.')
            for batch, (inputs, labels) in enumerate(self.train_dataloader):
                self.train_step(inputs, labels, batch, gradient_accumulation_steps, scheduler)
                # try:
                #     self.train_step(inputs, labels, batch, gradient_accumulation_steps, scheduler)
                # except Exception as e:
                #     error_print(e)
                #     warning_print('There is an EXCEPTION in a batch but train CONTINUE.')
            # except Exception as e:
            #     error_print(e)
            #     warning_print('There is an EXCEPTION in an epoch but train CONTINUE.')

            if (epoch + 1) % num_epochs_per_test == 0:
                # try:
                self.model.eval()
                self.eval()
                self.model.train()
                # except Exception as e:
                #     error_print(e)
                #     warning_print('There is an EXCEPTION in testing but train CONTINUE.')
                self.train_acc.append(100 * self.train_correct_num / self.train_check_num)
                self.train_check_num, self.train_correct_num = 0, 0

            if (epoch + 1) % num_epochs_per_plot == 0:
                self.plot_training_losses_per_batch()
                self.plot_training_losses_per_epoch()
                self.plot_accuracies(num_epochs_per_test)
                pass

            if (epoch + 1) > num_epochs * 0.75 and (epoch + 1) % num_epochs_per_save == 0:
                now = datetime.now()
                date_str = f'{now.year}-{now.month}-{now.day}-'
                torch.save(self.model.state_dict(), model_saving_path + f'{date_str}epoch{epoch + 1}.pt')
                running_print('Saved at', model_saving_path + f'{date_str}epoch{epoch + 1}.pt')

            if epoch == 0:
                avg_loss = sum(self.loss_per_batch) / len(self.loss_per_batch)
                num_one_epoch_loss_item = len(self.loss_per_batch)
            else:
                avg_loss = sum(self.loss_per_batch[-num_one_epoch_loss_item:]) / num_one_epoch_loss_item
            self.loss_per_epoch.append(avg_loss)
            running_print(f'Epoch {epoch + 1} end, avg loss: {avg_loss}.', '\n')
            pass

        running_print(f'Train end, last 5 testing acc: {self.accuracies[-5:]}')
        self.writer.close()
        pass

    @staticmethod
    def plot_process(*data_list_s, title=None, x_label=None, y_label=None, y_min=None, y_max=None, y_ticks=None,
                     y_decimals=None, x_scale=None, legend_s=None, grid=False, alpha_s=None, line_style_s=None,
                     line_width_s=None, color_s=None, marker_s=None, marker_size_s=None):
        """
        将一个或多个浮点数列表画成横轴为列表下标，纵轴为浮点数值的图，可设置图表各种属性

        :param data_list_s: 一个或多个浮点数列表
        :param title: 图表标题
        :param x_label: x轴标签
        :param y_label: y轴标签
        :param y_min: y轴最小值
        :param y_max: y轴最大值
        :param y_ticks: y轴刻度数值列表，可用于指定纵轴的标记
        :param y_decimals: y轴标记小数点位数
        :param x_scale: x轴缩放比例，如果不为None，则将横坐标整体乘以该值，以产生新的横坐标
        :param legend_s: 图例列表，一个字符串元素对应一个数据列表
        :param grid: 是否显示网格
        :param alpha_s: 不透明度
        :param line_style_s: 线条风格列表
        :param line_width_s: 线条宽度列表
        :param color_s: 线条颜色列表
        :param marker_s: 折线标记种类
        :param marker_size_s: 折线标记大小
        """
        if len(data_list_s) > 1:
            lengths = [len(l) for l in data_list_s]
            if len(set(lengths)) != 1:
                raise ValueError("所有的浮点数列表长度必须相同！")

        if len(data_list_s) == 1:
            plt.plot(data_list_s[0],
                     label=legend_s,
                     alpha=alpha_s if alpha_s is not None else 1,
                     linestyle=line_style_s if line_style_s is not None else '-',
                     linewidth=line_width_s if line_width_s is not None else 1,
                     color=color_s if color_s is not None else None,
                     marker=marker_s if marker_s is not None else None,
                     markersize=marker_size_s if marker_size_s is not None else None)
        else:
            for i, l in enumerate(data_list_s):
                legend = None if legend_s is None else legend_s[i]
                alpha = 1 if alpha_s is None else alpha_s[i]
                line_style = '-' if line_style_s is None else line_style_s[i]
                line_width = 1 if line_width_s is None else line_width_s[i]
                color = None if color_s is None else color_s[i]
                marker = None if marker_s is None else marker_s[i]
                marker_size = None if marker_size_s is None else marker_size_s[i]
                plt.plot(l, label=legend, alpha=alpha, linestyle=line_style, linewidth=line_width, color=color,
                         marker=marker, markersize=marker_size)

            plt.legend()

        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)

        if y_ticks is not None:
            plt.yticks(y_ticks)

        if y_decimals is not None:
            y_format = f"%.{y_decimals}f"
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: y_format % x))

        if x_scale is not None:
            plt.xticks(np.arange(len(data_list_s[0])), np.arange(len(data_list_s[0])) * x_scale)

        if title:
            plt.title(title)

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if legend_s is not None:
            plt.legend()

        if grid:
            plt.grid()

        plt.show(block=False)
        pass

    def plot_training_losses_per_batch(self):
        Trainer.plot_process(self.loss_per_batch, title=f'Train loss per batch * {self.save_print_loss_step}',
                             x_label='batch', y_label='loss', y_min=0.0, y_max=5.0,
                             y_ticks=[0.2, 1.0, 2.0, 3.0, 4.0],
                             y_decimals=2, legend_s='loss', grid=True, alpha_s=0.382, line_style_s='-',
                             line_width_s=0.382, color_s='navy', marker_s='.', marker_size_s=3)
        pass

    def plot_training_losses_per_epoch(self):
        Trainer.plot_process(self.loss_per_epoch, title='Train loss per epoch',
                             x_label='epoch', y_label='loss', y_min=0.0, y_max=4.0,
                             y_ticks=[0.2, 1.0, 2.0, 3.0],
                             y_decimals=2, legend_s='loss', grid=True, alpha_s=0.8, line_style_s='-',
                             line_width_s=1.1, color_s='maroon', marker_s='.', marker_size_s=3)
        pass

    def plot_accuracies(self, epochs_num_per_test):
        Trainer.plot_process(self.accuracies, self.train_acc, title=f'Accuracy per {epochs_num_per_test} epoch(s)',
                             x_label='epoch', y_label='acc', y_min=0, y_max=100,
                             y_ticks=[20, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100],
                             y_decimals=1, legend_s=['eval', 'train'], grid=True, alpha_s=[1, 1],
                             line_style_s=['-', '-'], line_width_s=[1.5, 1.5], color_s=['green', 'blue'],
                             x_scale=epochs_num_per_test, marker_s=['.', '.'], marker_size_s=[5, 5])
        pass
