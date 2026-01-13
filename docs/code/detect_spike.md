# 스파이크 검출 (Spike Detection) 원리

Kilosort4에서 스파이크 검출은 **템플릿 매칭(Template Matching)** 기반 알고리즘을 사용합니다. 이 문서는 스파이크 검출의 원리와 구현 코드 위치를 설명합니다.

## 전체 개요

Kilosort4의 스파이크 검출은 두 단계로 구성됩니다:

1. **Universal Template을 사용한 초기 검출**: 미리 정의된 또는 데이터에서 추출한 범용 템플릿을 사용하여 스파이크를 검출합니다.
2. **학습된 클러스터 파형을 사용한 재검출**: 초기 검출된 스파이크를 클러스터링한 후, 각 클러스터의 평균 파형을 템플릿으로 사용하여 더 정확하게 재검출합니다.

## 1단계: Universal Template을 사용한 초기 검출

### 원리

Universal template은 신경 스파이크의 일반적인 파형 특성을 나타내는 템플릿입니다. 이 단계에서는:

1. **템플릿 준비**: 
   - 데이터에서 직접 추출 (`templates_from_data=True`) 또는
   - 미리 계산된 템플릿 사용 (`templates_from_data=False`)

2. **템플릿 매칭**: 
   - 신호와 템플릿 간의 convolution을 수행하여 상관관계 계산
   - 여러 채널과 위치에 대해 가중치를 적용한 매칭 점수 계산

3. **피크 검출**:
   - 로컬 최대값을 찾기 위해 max pooling 적용
   - 임계값(`Th_universal`)을 넘는 피크를 스파이크로 판단

### 주요 코드 위치

**핵심 함수:**
- `kilosort/spikedetect.py`의 `template_match()` 함수 (133-178줄)
- `kilosort/spikedetect.py`의 `run()` 함수 (199-307줄)

**호출 경로:**
- `kilosort/run_kilosort.py`의 `detect_spikes()` 함수 (746줄)에서 호출
- `spikedetect.run()` (789줄) 실행

### 알고리즘 상세

#### 1. 템플릿과 신호의 Convolution

```python
# 템플릿을 사용하여 신호와 convolution 수행
W = ops['wTEMP'].unsqueeze(1)
B = conv1d(X.unsqueeze(1), W, padding=nt//2)
```

이 과정에서 각 템플릿이 신호의 각 시점에서 얼마나 잘 매칭되는지 계산합니다.

#### 2. 가중치를 적용한 매칭 점수 계산

```python
# 가까운 채널과 템플릿 위치에 가중치 적용
A = torch.einsum('ijk, jklm-> iklm', weigh, B[iC,:, nb*t:nb*(t+1)])
```

가중치(`weigh`)는 템플릿 위치와 채널 간의 거리에 따라 결정되며, 가까운 채널에 더 높은 가중치를 부여합니다.

#### 3. 로컬 피크 검출

```python
# 최대값 찾기
Aa, imax = torch.max(A.abs(), 0)

# 로컬 최대값을 찾기 위한 max pooling
Amaxs = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride=1, padding=nt0).squeeze(0)

# 임계값을 넘는 피크를 스파이크로 검출
xy = torch.logical_and(Amaxs==As, As > ops['Th_universal']).nonzero()
```

Max pooling을 통해 로컬 최대값을 찾고, 임계값을 넘는 지점을 스파이크로 판단합니다.

#### 4. PC Features 추출

검출된 각 스파이크에 대해 주변 채널의 파형을 추출하고, PCA를 적용하여 특징(feature)을 계산합니다:

```python
xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
xfeat = xsub @ ops['wPCA'].T
```

### 템플릿 준비 함수

**데이터에서 템플릿 추출:**
- `kilosort/spikedetect.py`의 `extract_wPCA_wTEMP()` 함수 (50-92줄)
- 단일 채널 임계값(`Th_single_ch`)을 사용하여 초기 스파이크 후보 추출
- 추출된 스니펫에 대해 PCA와 K-means 클러스터링을 수행하여 템플릿 생성

**미리 계산된 템플릿 사용:**
- `kilosort/spikedetect.py`의 `get_waves()` 함수 (94-98줄)
- 기본 템플릿 파일에서 로드

## 2단계: 학습된 클러스터 파형을 사용한 재검출

### 원리

1단계에서 검출된 스파이크를 클러스터링한 후, 각 클러스터의 평균 파형을 새로운 템플릿으로 사용하여 더 정확하게 재검출합니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/run_kilosort.py`의 `detect_spikes()` 함수 (746-865줄)
- `kilosort/template_matching.py`의 `extract()` 함수 (56-120줄)
- `kilosort/template_matching.py`의 `run_matching()` 함수 (170-243줄)

**실행 흐름:**
1. `detect_spikes()`에서 `spikedetect.run()` 호출 (789줄)
2. `clustering_qr.run()`으로 첫 번째 클러스터링 수행 (815줄)
3. `template_matching.postprocess_templates()`로 템플릿 후처리 (819줄)
4. `template_matching.extract()`로 학습된 템플릿으로 재검출 (842줄)

### 알고리즘 상세

#### 1. 클러스터 템플릿 생성

초기 검출된 스파이크를 클러스터링하여 각 클러스터의 평균 파형(`Wall`)을 계산합니다.

#### 2. 템플릿 매칭 재검출

학습된 템플릿을 사용하여 신호를 다시 스캔합니다:

```python
stt, amps, th_amps, Xres = run_matching(ops, X, U, ctc, device=device)
```

이 과정에서:
- 각 클러스터의 템플릿(`U`)과 신호를 매칭
- 임계값(`Th_learned`)을 넘는 피크를 검출
- 중첩된 스파이크도 처리 가능

#### 3. PC Features 업데이트

재검출된 스파이크에 대해 PC features를 업데이트합니다:

```python
xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
xfeat += amps * Ucc[:,stt[:,1]]
```

## 코드 흐름 요약

```
run_kilosort.py::detect_spikes()
    │
    ├─> spikedetect.run()  [1단계: Universal Template 검출]
    │       │
    │       ├─> extract_wPCA_wTEMP() 또는 get_waves()  [템플릿 준비]
    │       ├─> template_centers()  [템플릿 위치 계산]
    │       ├─> nearest_chans()  [가까운 채널 찾기]
    │       └─> template_match()  [각 배치에 대해 템플릿 매칭]
    │
    ├─> clustering_qr.run()  [첫 번째 클러스터링]
    │
    ├─> template_matching.postprocess_templates()  [템플릿 후처리]
    │
    └─> template_matching.extract()  [2단계: 학습된 템플릿으로 재검출]
            │
            └─> run_matching()  [템플릿 매칭 수행]
```

## 주요 파라미터

### 스파이크 검출 임계값

- `Th_universal` (기본값: 9): Universal template을 사용한 검출 시 임계값
- `Th_learned` (기본값: 8): 학습된 템플릿을 사용한 재검출 시 임계값
- `Th_single_ch` (기본값: 6): 단일 채널에서 초기 스니펫 추출 시 임계값

### 템플릿 관련 파라미터

- `n_templates` (기본값: 6): Universal template의 개수
- `n_pcs` (기본값: 3): PCA 특징의 차원 수
- `nearest_chans` (기본값: 10): 각 템플릿 위치에서 사용할 가까운 채널 수
- `nearest_templates` (기본값: 5): 각 위치에서 고려할 가까운 템플릿 수

### 공간 관련 파라미터

- `dmin`: 템플릿 중심의 수직 간격 (기본값: 자동 계산)
- `dminx`: 템플릿 중심의 수평 간격 (기본값: 32μm)
- `max_channel_distance`: 템플릿이 사용될 수 있는 최대 채널 거리 (기본값: 32μm)

## 핵심 아이디어

Kilosort4의 스파이크 검출 알고리즘의 핵심은:

1. **템플릿 기반 접근**: 단순한 임계값 검출이 아닌, 스파이크 파형의 공간-시간 패턴을 활용
2. **다중 채널 활용**: 여러 채널의 정보를 가중치로 결합하여 검출 정확도 향상
3. **반복적 개선**: Universal template으로 초기 검출 → 클러스터링 → 학습된 템플릿으로 재검출
4. **효율적인 계산**: GPU를 활용한 병렬 처리로 대용량 데이터 처리 가능

## 관련 파일

- `kilosort/spikedetect.py`: Universal template 기반 검출 구현
- `kilosort/template_matching.py`: 학습된 템플릿 기반 재검출 구현
- `kilosort/run_kilosort.py`: 전체 파이프라인 조율 및 `detect_spikes()` 함수
- `kilosort/clustering_qr.py`: 클러스터링 알고리즘
- `kilosort/parameters.py`: 기본 파라미터 설정
