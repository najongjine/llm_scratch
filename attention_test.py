import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 두 번째 입력 토큰이 쿼리입니다

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # 점곱 (1차원 벡터이므로 전치가 필요 없습니다)

# 정규화 안한 x^2 의 attention score: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
print(f"정규화 안한 x^2 의 attention score: {attn_scores_2}")

# 정규화한 x^2 어텐션 가중치: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print(" x^2 어텐션 가중치:", attn_weights_2)
print(" 합:", attn_weights_2.sum())

# tensor([0., 0., 0.])
context_vec_2 = torch.zeros(query.shape)
print(f"초기 문맥 벡터: {context_vec_2}") 
for i,x_i in enumerate(inputs):
    print(f"i:{i}")
    print(f"x_i:{x_i}")
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)


## 모든 입력 토큰에 대해 어텐션 가중치 계산하기
attn_scores = torch.empty(6, 6)

# attn_scores = inputs @ inputs.T # [6,3]@[3,6]=[6,6]  이거랑 같음
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(f" 모든 입력 토큰 끼리의 attention score: {attn_scores}")

attn_weights = torch.softmax(attn_scores, dim=-1)
print(f"정규화된 모든 입력 토큰 끼리의 attention score: {attn_weights}")

# @는 **행렬 곱셈 (Matrix Multiplication)**
all_context_vecs = attn_weights @ inputs # [6,6]@[6,3] = [6,3]
print(f"새로 나온 문맥 백터: {all_context_vecs}")