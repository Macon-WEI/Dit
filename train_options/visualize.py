import re
import matplotlib.pyplot as plt
import os


def visualize_log(log_file_path,img_save_path):
    # 读取日志文件内容
    # log_file_path = r"E:\同济\实验室\dif-acr-v2\021-DiT-L-4\log.txt"
    # log_file_path = r"test-log.txt"
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 提取日志文件中的损失值
    pattern = r'\(step=\d+\) Train Loss: (\d+\.\d+), Train Steps/Sec: \d+\.\d+'
    loss_values = re.findall(pattern, log_content)

    # 将损失值转换为浮点数
    loss_values = [float(loss) for loss in loss_values]

    # 生成训练步骤
    steps = list(range(1, len(loss_values) + 1))
    print(len(steps))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label='Train Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(img_save_path,'training_loss_curve.png'))
    # plt.show()