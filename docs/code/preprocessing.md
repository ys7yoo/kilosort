# 전처리 (Preprocessing) 원리

Kilosort4에서 전처리는 **고역 통과 필터링, Common Average Reference (CAR), 그리고 지역적 Whitening**을 통해 신호를 정제합니다. 이 문서는 전처리의 원리와 구현 코드 위치를 설명합니다.

## 전체 개요

Kilosort4의 전처리는 두 단계로 구성됩니다:

1. **전처리 파라미터 계산**: 고역 통과 필터와 Whitening 행렬을 미리 계산합니다.
2. **실제 데이터 필터링**: 각 배치를 로드할 때마다 순차적으로 필터링을 적용합니다.

## 1단계: 전처리 파라미터 계산

### 원리

전처리 단계에서 사용할 필터와 행렬을 미리 계산하여 저장합니다. 이렇게 하면 데이터를 처리할 때마다 반복 계산할 필요가 없습니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/run_kilosort.py`의 `compute_preprocessing()` 함수 (579-654줄)
- `kilosort/preprocessing.py`의 `get_highpass_filter()` 함수 (121-136줄)
- `kilosort/preprocessing.py`의 `get_whitening_matrix()` 함수 (96-119줄)

### 알고리즘 상세

#### 1. 고역 통과 필터 생성 (`get_highpass_filter`)

```python
# Butterworth 3차 고역 통과 필터 지정
b, a = butter(3, cutoff, fs=fs, btype='high')

# 임펄스 응답 계산
x = np.zeros(NT)
x[NT//2] = 1
hp_filter = filtfilt(b, a, x).copy()
```

- **Butterworth 3차 필터** 사용
- 기본 cutoff 주파수: **300 Hz**
- Fourier domain에서 적용하기 위해 임펄스 응답을 미리 계산

#### 2. Whitening 행렬 계산 (`get_whitening_matrix`)

**a) 공분산 행렬 계산:**

```python
# 일부 배치를 샘플링하여 공분산 계산
CC = torch.zeros((n_chan, n_chan), device=f.device)
k = 0
for j in range(0, f.n_batches-1, nskip):
    # 고역 통과 필터링된 데이터 로드
    X = f.padded_batch_to_torch(j)
    
    # 패딩 제거
    X = X[:, f.nt : -f.nt]
    
    # 누적 공분산 행렬
    CC = CC + (X @ X.T)/X.shape[1]
    k += 1

CC = CC / k  # 평균 공분산
```

**b) 지역적 Whitening 행렬 생성:**

```python
# 각 채널에 대해 지역적 whitening 행렬 계산
Wrot = whitening_local(CC, xc, yc, nrange=nrange, device=f.device)
```

각 채널 주변의 `nrange` 개 채널로 구성된 지역 공분산 행렬에서 whitening 행렬을 계산합니다.

## 2단계: 실제 데이터 필터링

### 원리

`BinaryFiltered` 클래스가 데이터를 로드할 때마다 자동으로 필터링을 적용합니다. 여러 단계를 순차적으로 수행하여 신호를 정제합니다.

### 주요 코드 위치

**핵심 함수:**
- `kilosort/io.py`의 `BinaryFiltered.filter()` 메서드 (990-1026줄)
- `kilosort/io.py`의 `BinaryFiltered.padded_batch_to_torch()` 메서드 (1037-1046줄)

### 필터링 단계

#### 1. 채널 선택

```python
# chanMap에 지정된 채널만 선택
if self.chan_map is not None:
    X = X[self.chan_map]
```

프로브 설정에서 지정한 채널만 사용합니다.

#### 2. 부호 반전 (옵션)

```python
if self.invert_sign:
    X = X * -1
```

데이터의 부호가 반대인 경우 사용합니다. 일반적으로는 필요하지 않습니다.

#### 3. 평균 제거 (DC 제거)

```python
X = X - X.mean(1).unsqueeze(1)
```

각 채널의 시간 평균을 제거하여 DC 성분을 제거합니다.

#### 4. Common Average Reference (CAR)

```python
if self.do_CAR:
    # remove the mean of each channel, and the median across channels
    X = X - torch.median(X, 0)[0]
```

모든 채널의 **중앙값(median)**을 빼서 공통 노이즈를 제거합니다. 평균 대신 중앙값을 사용하는 이유는 outlier의 영향을 줄이기 위함입니다.

**권장 설정**: `do_CAR=True` (기본값)

#### 5. 고역 통과 필터링

```python
# high-pass filtering in the Fourier domain (much faster than filtfilt etc)
if self.hp_filter is not None:
    fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
    X = torch.real(ifft(fft(X) * torch.conj(fwav)))
    X = fftshift(X, dim = -1)
```

- **Fourier domain에서 필터링**: 시간 도메인의 `filtfilt`보다 훨씬 빠릅니다.
- 미리 계산된 필터를 Fourier domain으로 변환하여 적용합니다.

#### 6. Artifact 제거

```python
if self.artifact_threshold < np.inf:
    if torch.any(torch.abs(X) >= self.artifact_threshold):
        # Assume the batch contains a recording artifact.
        # Skip subsequent preprocessing, zero-out the batch.
        return torch.zeros_like(X)
```

임계값을 넘는 큰 신호가 있는 배치는 recording artifact로 간주하고 제로화합니다.

#### 7. Whitening (선택적 drift 보정 포함)

```python
# whitening, with optional drift correction
if self.whiten_mat is not None:
    if self.dshift is not None and ops is not None and ibatch is not None:
        M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
        X = (M @ self.whiten_mat) @ X
    else:
        X = self.whiten_mat @ X
```

- **지역적 Whitening**: 각 채널 주변의 인접 채널만 사용하여 공간적 구조를 보존합니다.
- **Drift 보정 통합**: Drift 보정이 계산된 경우, drift 행렬을 먼저 적용한 후 whitening을 수행합니다.

## Whitening 행렬 계산 상세

### 지역적 Whitening (`whitening_local`)

각 채널에 대해 가장 가까운 채널들로 구성된 지역 공분산 행렬에서 whitening 행렬을 계산합니다:

```python
def whitening_local(CC, xc, yc, nrange=32, device=torch.device('cuda')):
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan, Nchan), device=device)
    
    for j in range(CC.shape[0]):
        # 가장 가까운 nrange 개 채널 선택
        ds = (xc[j] - xc)**2 + (yc[j] - yc)**2
        isort = np.argsort(ds)
        ix = isort[:nrange]
        
        # 지역 공분산 행렬에서 whitening 행렬 계산
        wrot = whitening_from_covariance(CC[np.ix_(ix, ix)])
        
        # 중심 채널에 대한 whitening 벡터 저장
        Wrot[j, ix] = wrot[0]
    
    return Wrot
```

### ZCA Whitening (`whitening_from_covariance`)

ZCA (Zero Component Analysis) whitening을 사용합니다:

```python
def whitening_from_covariance(CC):
    # SVD를 사용하여 whitening 행렬 계산
    E, D, V = torch.linalg.svd(CC)
    eps = 1e-6
    Wrot = (E / (D+eps)**.5) @ E.T
    return Wrot
```

ZCA whitening은 공분산 행렬을 단위 행렬로 변환하면서 원래 좌표계를 최대한 보존합니다.

## Drift 보정 통합

Drift 보정이 활성화된 경우, whitening 단계에서 drift 행렬을 함께 적용합니다:

```python
# Drift 행렬 계산
M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)

# Drift 보정 + Whitening
X = (M @ self.whiten_mat) @ X
```

`get_drift_matrix()`는 채널 위치의 drift를 보정하기 위한 보간 행렬을 생성합니다.

## 코드 흐름 요약

```
run_kilosort.py::_sort()
    │
    └─> compute_preprocessing()  [전처리 파라미터 계산]
            │
            ├─> preprocessing.get_highpass_filter()  [고역 통과 필터 생성]
            │       │
            │       └─> butter() + filtfilt()  [Butterworth 필터 임펄스 응답]
            │
            ├─> io.BinaryFiltered()  [필터링 가능한 파일 객체 생성]
            │
            └─> preprocessing.get_whitening_matrix()  [Whitening 행렬 계산]
                    │
                    ├─> BinaryFiltered.padded_batch_to_torch()  [배치 로드]
                    │       └─> BinaryFiltered.filter()  [필터링 적용]
                    │               │
                    │               ├─> 채널 선택
                    │               ├─> 부호 반전 (옵션)
                    │               ├─> 평균 제거
                    │               ├─> CAR (옵션)
                    │               ├─> 고역 통과 필터링
                    │               ├─> Artifact 제거
                    │               └─> Whitening
                    │
                    └─> preprocessing.whitening_local()  [지역적 whitening]
                            │
                            └─> preprocessing.whitening_from_covariance()  [ZCA whitening]
    │
    └─> compute_drift_correction()  [Drift 보정 계산]
    │
    └─> 이후 단계에서 BinaryFiltered 사용 시 자동으로 필터링 적용
```

## 주요 파라미터

### 필터링 파라미터

- `highpass_cutoff` (기본값: 300): 고역 통과 필터 cutoff 주파수 (Hz)
- `do_CAR` (기본값: True): Common Average Reference 적용 여부
  - `True`: 모든 채널의 중앙값을 빼서 공통 노이즈 제거 (권장)
  - `False`: CAR 적용 안 함
- `invert_sign` (기본값: False): 데이터 부호 반전 여부
  - 일반적으로는 필요하지 않음
- `artifact_threshold` (기본값: inf): Artifact 검출 임계값
  - 이 값을 넘는 신호가 있는 배치는 제로화됨

### Whitening 파라미터

- `whitening_range` (기본값: 32): 지역 whitening에 사용할 채널 범위 (μm)
  - 각 채널 주변의 이 거리 내 채널들만 사용
- `nskip` (기본값: 25): 공분산 계산 시 배치 샘플링 간격
  - 모든 배치를 사용하지 않고 일부만 샘플링하여 계산 속도 향상

### 데이터 관련 파라미터

- `chanMap`: 사용할 채널 인덱스 배열
  - 프로브 설정에서 지정
- `tmin`, `tmax`: 처리할 시간 범위 (초)
  - `None`이면 전체 데이터 처리
- `batch_downsampling`: 배치 다운샘플링 비율
  - 메모리 절약을 위해 사용

## 핵심 아이디어

Kilosort4의 전처리 알고리즘의 핵심은:

1. **지역적 Whitening**: 전체 채널이 아닌 인접 채널만 사용하여 공간적 구조를 보존하면서 노이즈를 제거
2. **Fourier Domain 필터링**: 시간 도메인의 `filtfilt`보다 훨씬 빠른 처리 속도
3. **Median-based CAR**: 평균 대신 중앙값을 사용하여 outlier의 영향을 줄임
4. **Drift 보정 통합**: Whitening 단계에서 drift 보정 행렬을 함께 적용하여 효율성 향상
5. **온라인 처리**: 각 배치를 로드할 때마다 자동으로 필터링이 적용되어 메모리 효율적

## 관련 파일

- `kilosort/preprocessing.py`: 전처리 필터 생성 함수들
  - `get_highpass_filter()`: 고역 통과 필터 생성
  - `get_whitening_matrix()`: Whitening 행렬 계산
  - `whitening_local()`: 지역적 whitening 행렬 생성
  - `whitening_from_covariance()`: ZCA whitening 계산
  - `get_drift_matrix()`: Drift 보정 행렬 생성
- `kilosort/io.py`: `BinaryFiltered` 클래스
  - `filter()`: 실제 필터링 수행
  - `padded_batch_to_torch()`: 배치 로드 및 필터링
- `kilosort/run_kilosort.py`: `compute_preprocessing()` 함수
- `kilosort/parameters.py`: 기본 파라미터 설정
