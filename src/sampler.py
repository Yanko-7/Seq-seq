import torch
import torch.nn.functional as F


class Sampler:
    def __init__(self, model, tokenizer, temperature=1.0, top_k=50):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k

    @torch.inference_mode()  # 比 no_grad 更极速的推理模式
    def generate(self, max_new_tokens: int, prompt_tokens=None):
        idx = prompt_tokens if prompt_tokens is not None else torch.zeros(...)

        for _ in range(max_new_tokens):
            logits = self.model(idx)
            logits = logits[:, -1, :] / self.temperature
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k)
            # 从 Top-K 中采样
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            # 获取对应的 token ID
            next_token = top_k_indices.gather(-1, next_token)
            idx = torch.cat((idx, next_token), dim=1)

        # 2. 拿到完整的 Token 序列后，调用 VQ-VAE 解码
        generated_geometry = self.tokenizer.decode(idx)
        return generated_geometry
