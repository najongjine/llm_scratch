import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] # 두 번째 입력 원소
d_in = inputs.shape[1] # 입력 임베딩 크기, d=3
d_out = 2 # 출력 임베딩 크기, d=2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# @ matrix muliplication
query_2 = x_2 @ W_query # 두 번째 입력 원소에 대한 값을 계산하므로 _2로 씁니다.
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(f"query_2: {query_2}")

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)