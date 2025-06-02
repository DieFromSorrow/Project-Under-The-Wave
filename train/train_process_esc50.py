import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from datasets import ESC50Dataset
from models.eca_res_net_mfcc import ECAResNetMfccClassifier  # 替换为你的模型定义


def cross_validate(
        root_dir,
        model_class,
        model_save_path,
        n_epochs=10,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    # 存储每折的结果
    fold_results = {
        'val_acc': [],
        'val_loss': [],
        'train_history': []  # 新增：存储训练历史
    }

    # 创建模型保存目录
    save_dir = Path(model_save_path)
    save_dir.mkdir(exist_ok=True)

    # 5折交叉验证
    for fold in [5]:
        print(f"\n=== Fold {fold}/5 ===")

        # 初始化数据集（当前fold作为验证集）
        train_dataset = ESC50Dataset(
            root_dir=root_dir,
            folds=[f for f in range(1, 6) if f != fold],  # 其他4个fold训练
            sample_rate=22050,
            n_mels=64
        )

        val_dataset = ESC50Dataset(
            root_dir=root_dir,
            folds=[fold],  # 当前fold验证
            sample_rate=22050,
            n_mels=64
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        # 初始化模型和优化器
        model = model_class(in_channels=64, num_classes=len(train_dataset.get_class_names()),
                            dropout=0.2, layers=[2, 2, 6, 2], r=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        val_acc, val_loss = 0, 0
        best_val_acc = 0.0

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        # 训练循环
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0

            for X, y, _, _ in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证评估
            val_acc, val_loss = evaluate(model, val_loader, device, criterion)

            # 记录历史数据
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {train_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            if epoch >= (n_epochs * 3 // 4) and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_dir / f"fold{fold}_epoch{epoch}_acc{val_acc:.2f}.pth")
                print(f"Model saved at epoch {epoch} with val acc {val_acc:.2f}%")

        # 记录当前fold的最终结果
        fold_results['val_acc'].append(val_acc)
        fold_results['val_loss'].append(val_loss)
        fold_results['train_history'].append(history)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold} Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Val Accuracy', color='green')
        plt.title(f'Fold {fold} Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"../plots/fold_{fold}_training_curve.png")
        plt.close()

    # 打印最终交叉验证结果
    print("\n=== Cross-Validation Results ===")
    print(f"Mean Val Acc: {np.mean(fold_results['val_acc']):.2f}% "
          f"(±{np.std(fold_results['val_acc']):.2f})")
    print(f"Mean Val Loss: {np.mean(fold_results['val_loss']):.4f}")


def evaluate(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for X, y, _, _ in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(dataloader)
    return accuracy, avg_loss


if __name__ == "__main__":
    cross_validate(
        root_dir="../data/ESC-50/ESC-50-master",
        model_class=ECAResNetMfccClassifier,
        model_save_path='../checkpoints/esc50',
        n_epochs=100,
        batch_size=128
    )
