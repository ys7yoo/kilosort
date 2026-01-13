# 클러스터링 (Clustering) 원리

Kilosort4에서 클러스터링은 **그래프 기반 알고리즘**을 사용하여 검출된 스파이크를 뉴런 단위로 그룹화합니다. 이 문서는 클러스터링의 원리와 구현 코드 위치를 설명합니다.

## 전체 개요

Kilosort4의 클러스터링은 여러 단계로 구성됩니다:

1. **공간적 그룹화**: 프로브의 x-y 위치를 기반으로 템플릿을 그룹화합니다.
2. **그래프 기반 클러스터링**: FAISS를 사용하여 이웃 행렬을 생성하고, 반복적 할당을 통해 클러스터를 형성합니다.
3. **Hierarchical 병합**: 클러스터 간 유사도를 기반으로 병합 트리를 생성합니다.
4. **Swarm Splitting**: 이중봉(bimodal) 분포와 refractory period를 기반으로 클러스터를 분할합니다.
5. **최종 병합**: 유사한 클러스터를 템플릿 매칭으로 병합합니다.

## 1단계: 공간적 그룹화

### 원리

프로브의 공간적 위치(x, y 좌표)를 기반으로 템플릿을 그룹화합니다. 각 그룹에 대해 독립적으로 클러스터링을 수행하여 계산 효율성을 높입니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/clustering_qr.py`의 `run()` 함수 (416-557줄)
- `kilosort/clustering_qr.py`의 `x_centers()` 함수 (338-384줄)
- `kilosort/clustering_qr.py`의 `y_centers()` 함수 (387-395줄)
- `kilosort/clustering_qr.py`의 `get_nearest_centers()` 함수 (398-413줄)

### 알고리즘 상세

#### 1. 중심점 계산

```python
# x축 중심점: 히스토그램 피크 또는 k-means로 계산
xcent = x_centers(ops)

# y축 중심점: dmin 간격으로 균등 분할
ycent = y_centers(ops)
```

각 템플릿은 가장 가까운 (x, y) 중심점에 할당됩니다.

#### 2. 그룹별 데이터 추출

```python
Xd, igood, ichan = get_data_cpu(
    ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
    dmin=dmin, dminx=dminx, ix=ix,
)
```

각 중심점 주변의 스파이크 PC features를 추출하여 클러스터링에 사용합니다.

## 2단계: 그래프 기반 클러스터링

### 원리

각 공간 그룹 내에서 그래프 기반 클러스터링을 수행합니다. 이웃 행렬(neighbor matrix)을 생성하고, 반복적으로 클러스터 할당을 업데이트합니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/clustering_qr.py`의 `cluster()` 함수 (121-180줄)
- `kilosort/clustering_qr.py`의 `neigh_mat()` 함수 (20-66줄)
- `kilosort/clustering_qr.py`의 `kmeans_plusplus()` 함수 (183-278줄)
- `kilosort/clustering_qr.py`의 `assign_iclust()` 함수 (69-86줄)
- `kilosort/clustering_qr.py`의 `assign_isub()` 함수 (89-105줄)

### 알고리즘 상세

#### 1. 이웃 행렬 생성 (`neigh_mat`)

FAISS를 사용하여 각 스파이크의 k-최근접 이웃을 찾습니다:

```python
# 다운샘플링된 서브셋 생성
Xsub = Xd[::nskip]

# FAISS로 L2 거리 기반 이웃 검색
index = faiss.IndexFlatL2(dim)
index.add(Xsub)
_, kn = index.search(Xd, n_neigh)

# 희소 인접 행렬 생성
M = csr_matrix((dexp.flatten(), (rows, kn.flatten())), 
               (kn.shape[0], n_nodes))
```

`kn`은 각 스파이크의 이웃 인덱스를 포함하고, `M`은 인접 행렬입니다.

#### 2. 초기 클러스터 할당 (`kmeans_plusplus`)

K-means++ 알고리즘을 사용하여 초기 클러스터 중심을 선택합니다:

```python
# 설명되지 않은 분산을 기반으로 후보 중심 샘플링
v2 = torch.relu(vtot - vexp0)
isamp = torch.multinomial(v2, ntry)

# 각 후보에 대해 설명되는 분산 계산
vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
dexp = torch.relu(vexp - vexp0.unsqueeze(1))

# 가장 많은 분산을 설명하는 중심 선택
imax = torch.argmax(vsum)
```

이 알고리즘은 초기 클러스터 중심을 잘 분산시켜 클러스터링 품질을 향상시킵니다.

#### 3. 반복적 클러스터 할당 업데이트

두 단계를 반복합니다:

**a) 서브셋 할당 업데이트 (`assign_isub`):**
```python
# 각 서브셋 스파이크를 가장 많은 이웃이 속한 클러스터에 할당
xS = coo(iis, tones2.flatten(), (nsub, nclust))
isub = torch.argmax(xS, 1)
```

**b) 전체 스파이크 할당 업데이트 (`assign_iclust`):**
```python
# 각 스파이크를 이웃 서브셋 스파이크들이 속한 클러스터에 할당
xN = coo(ij, tones2.flatten(), (n_spikes, nclust))
iclust = torch.argmax(xN, 1)
```

이 과정을 `niter`번 반복하여 클러스터 할당을 최적화합니다.

## 3단계: Hierarchical 병합

### 원리

클러스터 간 유사도를 기반으로 병합 트리(dendrogram)를 생성합니다. 이 트리는 이후 분할 단계에서 사용됩니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/hierarchical.py`의 `maketree()` 함수 (93-104줄)
- `kilosort/hierarchical.py`의 `merge_reduce()` 함수 (30-43줄)
- `kilosort/hierarchical.py`의 `find_merges()` 함수 (45-82줄)

### 알고리즘 상세

#### 1. 클러스터 간 연결 행렬 계산

```python
# 클러스터-스파이크 할당 행렬
q = csr_matrix((np.ones(NN,), (iclust, np.arange(NN))), (nc, NN))
r = csr_matrix((np.ones(nr,), (np.arange(nr), iclust0)), (nr, nc))

# 클러스터 간 실제 연결 수
cc = (q @ M @ r).toarray()

# 기대 연결 수 (null model)
cneg = .001 + np.outer(q @ ki , kj @ r)/m
```

#### 2. 병합 비율 계산 및 병합

```python
# 실제 연결 수 대 기대 연결 수의 비율
crat = cc/cneg

# 가장 높은 비율을 가진 클러스터 쌍을 병합
y, x = np.unravel_index(np.argmax(crat), cc.shape)
```

이 과정을 모든 클러스터가 하나로 병합될 때까지 반복하여 병합 트리를 생성합니다.

## 4단계: Swarm Splitting

### 원리

병합 트리를 역순으로 탐색하며, 이중봉 분포와 refractory period를 기반으로 클러스터를 분할합니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/swarmsplitter.py`의 `split()` 함수 (80-132줄)
- `kilosort/swarmsplitter.py`의 `check_split()` 함수 (11-29줄)
- `kilosort/swarmsplitter.py`의 `bimod_score()` 함수 (40-51줄)
- `kilosort/swarmsplitter.py`의 `refractoriness()` 함수 (62-78줄)

### 알고리즘 상세

#### 1. 분할 기준 평가

네 가지 기준을 순차적으로 평가합니다:

**a) 전역 모듈성 (Global Modularity):**
```python
if tstat[kk,0] < 0.2:
    criterion = -1  # 분할하지 않음
```

**b) Refractory Period:**
```python
is_refractory = check_CCG(st1, st2)[1]
if is_refractory:
    criterion = 1  # 분할하지 않음
```

**c) 이중봉 분포 (Bimodality):**
```python
xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
criterion = 2 * (score < .6) - 1
```

`check_split()`은 가중치 최소제곱법으로 분리 초평면을 찾고, 투영된 분포의 이중봉 정도를 평가합니다.

**d) 지역 모듈성 (Local Modularity):**
```python
score = tstat[kk,-1]
criterion = score > .15
```

#### 2. 분할 실행

```python
if criterion==1:
    valid_merge[kk] = 0
    clean_tree(valid_merge, xtree, xtree[kk,0])
    clean_tree(valid_merge, xtree, xtree[kk,1])
```

분할이 결정되면 해당 병합을 무효화하고, 하위 노드들도 재귀적으로 정리합니다.

#### 3. 새로운 클러스터 할당

```python
iclust = swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)
```

유효한 병합만 남은 트리에서 최종 클러스터 할당을 생성합니다.

## 5단계: 최종 병합

### 원리

템플릿 간 상관관계와 CCG(Cross-Correlogram)를 기반으로 유사한 클러스터를 병합합니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/template_matching.py`의 `merging_function()` 함수 (247-340줄)
- `kilosort/CCG.py`의 `check_CCG()` 함수

### 알고리즘 상세

#### 1. 템플릿 상관관계 계산

```python
# 정규화된 템플릿 간 상관관계
UtU = torch.einsum('lk, jlm -> jkm', Wnorm[kk], Wnorm)
ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)
cmax, imax = ctc.max(1)
```

#### 2. CCG 기반 병합 검증

```python
# 상관관계가 높은 클러스터 쌍에 대해 CCG 검사
st0 = st[:,0][clu2==kk] / ops['fs']
st1 = st[:,0][clu2==jj] / ops['fs']
_, is_ccg, _ = CCG.check_CCG(st0, st1, 
                             acg_threshold=acg_threshold,
                             ccg_threshold=ccg_threshold)
```

CCG가 refractory period를 보이면 같은 뉴런에서 나온 것으로 판단하여 병합합니다.

#### 3. 시간 지연 보정

병합 시 두 클러스터 간의 시간 지연을 보정합니다:

```python
dt = (imax[kk] - imax[jj]).item()
if dt != 0 and check_dt:
    # 스파이크 시간과 features를 지연만큼 이동
    st[idx,0] -= dt
    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
```

## 코드 흐름 요약

```
run_kilosort.py::cluster_spikes()
    │
    └─> clustering_qr.run()  [공간 그룹화 및 클러스터링]
            │
            ├─> x_centers() / y_centers()  [공간 중심점 계산]
            ├─> get_nearest_centers()  [템플릿 그룹화]
            │
            └─> for each spatial group:
                    │
                    ├─> get_data_cpu()  [그룹별 데이터 추출]
                    │
                    ├─> cluster()  [그래프 기반 클러스터링]
                    │       │
                    │       ├─> neigh_mat()  [FAISS로 이웃 행렬 생성]
                    │       ├─> kmeans_plusplus()  [초기 클러스터 할당]
                    │       └─> assign_iclust() / assign_isub()  [반복적 할당]
                    │
                    ├─> hierarchical.maketree()  [병합 트리 생성]
                    │       │
                    │       ├─> prepare()  [클러스터 간 연결 행렬 계산]
                    │       └─> merge_reduce()  [병합 트리 생성]
                    │
                    └─> swarmsplitter.split()  [클러스터 분할]
                            │
                            ├─> check_split()  [이중봉 분포 검사]
                            ├─> refractoriness()  [Refractory period 검사]
                            └─> new_clusters()  [최종 클러스터 할당]
    │
    └─> template_matching.merging_function()  [최종 병합]
            │
            └─> CCG.check_CCG()  [CCG 기반 병합 검증]
```

## 주요 파라미터

### 클러스터링 파라미터

- `cluster_downsampling` (기본값: 1): 이웃 검색 시 다운샘플링 비율
- `cluster_neighbors` (기본값: 10): 각 스파이크의 이웃 수
- `max_cluster_subset` (기본값: 25000): 이웃 검색에 사용할 최대 서브셋 크기
- `cluster_init_seed` (기본값: 1): K-means++ 초기화 시드
- `nclust` (기본값: 200): 초기 클러스터 수
- `niter` (기본값: 200): 클러스터 할당 반복 횟수

### 공간 관련 파라미터

- `dmin`: 템플릿 중심의 수직 간격 (기본값: 자동 계산)
- `dminx`: 템플릿 중심의 수평 간격 (기본값: 32μm)
- `x_centers`: x축 중심점 (기본값: 자동 계산)

### 분할 관련 파라미터

- `bimodality_threshold` (기본값: 0.6): 이중봉 분포 검사 임계값
- `modularity_threshold` (기본값: 0.2): 전역 모듈성 임계값
- `local_modularity_threshold` (기본값: 0.15): 지역 모듈성 임계값

### 병합 관련 파라미터

- `r_thresh` (기본값: 0.5): 템플릿 상관관계 병합 임계값
- `acg_threshold` (기본값: 0.1): Auto-correlogram 임계값
- `ccg_threshold` (기본값: 0.25): Cross-correlogram 임계값

## 핵심 아이디어

Kilosort4의 클러스터링 알고리즘의 핵심은:

1. **그래프 기반 접근**: 스파이크 간 유사도를 그래프로 표현하고, 이웃 정보를 활용하여 클러스터링
2. **공간적 분할**: 프로브의 공간적 구조를 활용하여 계산 효율성 향상
3. **Hierarchical 구조**: 병합 트리를 생성하여 다양한 해상도의 클러스터 구조 제공
4. **생물학적 제약 활용**: Refractory period와 이중봉 분포를 활용하여 클러스터 품질 향상
5. **반복적 개선**: 분할과 병합을 반복하여 최적의 클러스터 할당 도출

## 관련 파일

- `kilosort/clustering_qr.py`: 핵심 클러스터링 알고리즘 (그래프 기반 클러스터링, 이웃 행렬 생성)
- `kilosort/hierarchical.py`: 병합 트리 생성 및 관리
- `kilosort/swarmsplitter.py`: 클러스터 분할 알고리즘 (이중봉 분포, refractory period 검사)
- `kilosort/template_matching.py`: 최종 병합 함수 (템플릿 상관관계, CCG 기반 병합)
- `kilosort/CCG.py`: Cross-correlogram 계산 및 분석
- `kilosort/run_kilosort.py`: 전체 파이프라인 조율 및 `cluster_spikes()` 함수
- `kilosort/parameters.py`: 기본 파라미터 설정
