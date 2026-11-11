"""[cls-scale-decomp] 순서 k 파일 만들기 (DECOMP 부분 5절).

사용: python scripts/decomp/make_order.py <순서 1 parquet> <k> <출력 parquet>
- 팀 규칙(판정자 실험 H): 순서 k = 순서 1 파일을 np.random.default_rng(1000 + k).permutation(n) 으로 섞은 것. k = 1이면 그대로.
- 지문 = sha1(','.join(id))[:12]. LmArena 순서 1 = 1f38e5aad950, 순서 2 = e895e59995c3 (팀 기록과 대조용)
"""
import hashlib
import sys

import numpy as np
import pyarrow.parquet as pq


def fingerprint(table) -> str:
    ids = [str(x) for x in table.column("id").to_pylist()]
    return hashlib.sha1(",".join(ids).encode()).hexdigest()[:12]


def main():
    src, k, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    t = pq.read_table(src)
    if k != 1:
        t = t.take(np.random.default_rng(1000 + k).permutation(t.num_rows))
    pq.write_table(t, dst)
    print(f"order {k}: {t.num_rows} rows, fingerprint {fingerprint(t)} -> {dst}")


if __name__ == "__main__":
    main()
