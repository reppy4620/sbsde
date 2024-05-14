import torch
import torch.nn as nn
import torch.optim as optim


# シンプルなネットワークの定義
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# ハイパーパラメータの設定
input_dim = 10
output_dim = 1
learning_rate = 0.01
grad_penalty_weight = 0.1  # 勾配ペナルティの係数

# モデル、損失関数、オプティマイザの定義
u = SimpleModel(input_dim, output_dim)
s = SimpleModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer_u = optim.SGD(u.parameters(), lr=learning_rate)
optimizer_s = optim.SGD(s.parameters(), lr=learning_rate)

# ダミーデータの作成
X = torch.randn(
    100, input_dim, requires_grad=True
)  # 勾配計算を有効にするためにrequires_grad=True
y_u = torch.randn(100, output_dim)
y_s = torch.randn(100, output_dim)

# トレーニングループ
num_epochs = 100
for epoch in range(num_epochs):
    u.train()
    s.train()

    optimizer_u.zero_grad()
    optimizer_s.zero_grad()

    # フォワードパス
    outputs_u = u(X)
    outputs_s = s(X)
    loss_u = criterion(outputs_u, y_u)
    loss_s = criterion(outputs_s, y_s)

    # 勾配ペナルティの計算
    grads_u = torch.autograd.grad(
        outputs=outputs_u,
        inputs=X,
        grad_outputs=torch.ones_like(outputs_u),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grads_u is not None:
        grad_penalty_u = (grads_u.norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty_u = torch.tensor(0.0)

    grads_s = torch.autograd.grad(
        outputs=outputs_s,
        inputs=X,
        grad_outputs=torch.ones_like(outputs_s),
        create_graph=True,
        allow_unused=True,
    )[0]
    if grads_s is not None:
        grad_penalty_s = (grads_s.norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty_s = torch.tensor(0.0)

    # 勾配ペナルティを損失に追加
    total_loss_u = loss_u + grad_penalty_weight * grad_penalty_u
    total_loss_s = loss_s + grad_penalty_weight * grad_penalty_s

    # バックプロパゲーションとオプティマイゼーション
    total_loss_u.backward(retain_graph=True)  # 計算グラフを保持
    total_loss_s.backward()  # 最後のバックプロパゲーションでは計算グラフの保持は不要

    optimizer_u.step()
    optimizer_s.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss U: {total_loss_u.item():.4f}, Grad Penalty U: {grad_penalty_u.item():.4f}"
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss S: {total_loss_s.item():.4f}, Grad Penalty S: {grad_penalty_s.item():.4f}"
        )

print("Training complete.")
