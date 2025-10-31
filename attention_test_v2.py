import torch
import torch.nn as nn

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

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        #단순한 '의미'를 3가지 '관점'으로 분리
        #질문하는 역할(Q). "다른 단어에게 무엇을 물어볼지"
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        #정보를 제공하는 역할(K). "다른 단어의 질문에 어떤 정보를 열쇠처럼 제공할지"
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        #실제 의미를 전달하는 역할(V). 회의 결과로 전달할 실제 의미 정보
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key # 모든 단어(x^1~x^i)의 '열쇠' 정보 (K)
        queries = x @ self.W_query # 모든 단어의 '질문' 정보 (Q)
        values = x @ self.W_value # 모든 단어의 '의미 값' 정보 (V)

        #각 단어의 '질문'이 다른 단어의 '열쇠'와 얼마나 잘 맞는지 계산. Q가 **"내가 지금 찾는 정보"**라면, K는 **"내가 가진 정보"**입니다.
        attn_scores = queries @ keys.T # omega

        #**'journey'**가 다른 단어들(Your, starts, step 등)에게 보낼 총 100%의 집중력을 얼마나 나눠서 줄지 결정
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        #문장 속 모든 관련 단어의 정보가 가장 적절한 비율로 섞여 담겨있습니다. 가장 중요한 정보만 쏙 뽑아와서 새로운 의미를 만들기 위해서
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

## 코잘 어텐션 마스크 적용하기 ##
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

## 미래 단어를 참조하지 못하도록 마스킹 처리
context_length = attn_scores.shape[0]

# Upper triangular matrix 생성
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)