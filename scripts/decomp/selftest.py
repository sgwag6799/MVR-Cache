"""[cls-scale-decomp] RunPod에서 본 실행 전에 돌리는 점검 (모델 없이 1초, --with-model이면 BGE로 수 초).

사용: cd mvr-cache && python ../scripts/decomp/selftest.py [--with-model]
1) 분할 0 + CLS 포함 조각 벡터 = 전체 토큰 평균(바닐라 임베딩), CLS 제외 = 1번 토큰부터 평균
2) 자르는 지점이 있을 때 첫 조각만 [CLS]부터 시작
3) score_scale raw = 코사인, unit = (코사인+1)/2
4) (--with-model) 실제 BGE로 P1 경로 벡터와 바닐라 EmbeddingModel.get_embedding 벡터의 코사인 ≈ 1
"""
import sys
import torch
import torch.nn.functional as F

from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.vcache_core.splitter.PrefixRuleSplitter import PrefixRuleSplitter
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy as P

torch.manual_seed(0)
L, H = 12, 8
tok = torch.randn(L, H)

sent, _ = MaxSimSplitter._segment_embeds_from_pointers(tok, L, [])
assert torch.allclose(sent[0], tok[1:L].mean(0), atol=1e-6), "CLS 제외 조각이 1번 토큰부터가 아니다"
assert torch.allclose(PrefixRuleSplitter._first_segment_with_cls(tok, L, []), tok[0:L].mean(0), atol=1e-6)
assert torch.allclose(PrefixRuleSplitter._first_segment_with_cls(tok, L, [3, 7]), tok[0:4].mean(0), atol=1e-6)
sent2, _ = MaxSimSplitter._segment_embeds_from_pointers(tok, L, [3, 7])
assert torch.allclose(sent2[0], tok[1:4].mean(0), atol=1e-6) and torch.allclose(sent2[1], tok[4:8].mean(0), atol=1e-6)

a, b = torch.randn(1, H), torch.randn(1, H)
cos = float(F.cosine_similarity(a, b).item())
assert abs(P._maxsim_from_tensors(a, b, scale="raw") - cos) < 1e-6
assert abs(P._maxsim_from_tensors(a, b, scale="unit") - (cos + 1) / 2) < 1e-6
assert abs(P._maxsim_from_tensors(a, b) - (cos + 1) / 2) < 1e-6  # 기본값은 원본과 같다
print("selftest 1~3 OK")

if "--with-model" in sys.argv:
    from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = EmbeddingModel(device=dev)
    texts = ["What is the capital of France?", "Classify the sentiment: this movie was great, truly.", "hi"]
    for inc in (True, False):
        sp = PrefixRuleSplitter(device=dev, embedding_model=em, max_cuts=0, cap_scope="all", include_cls=inc)
        for t in texts:
            v = sp.split_text_return_maxsim_tensor(t)
            assert v.shape[0] == 1, "분할 0인데 조각이 1개가 아니다"
            van = torch.tensor(em.get_embedding(t)).float().to(v.device)
            c = float(F.cosine_similarity(v[0:1], van.unsqueeze(0)).item())
            if inc:
                assert c > 0.99999, f"CLS 포함 벡터가 바닐라와 다르다: cos={c}"
            print(f"include_cls={inc} cos(분할0 벡터, 바닐라)={c:.6f}  {t[:30]!r}")
    print("selftest 4 OK (CLS 포함 = 바닐라 벡터, CLS 제외는 위 코사인만큼 다름)")
