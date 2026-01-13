# plots.py 코드 설명

이 파일은 Kilosort4 스파이크 정렬 파이프라인에서 생성되는 진단 및 결과 플롯을 생성하는 모듈입니다. matplotlib을 사용하여 4가지 주요 플롯을 생성합니다.

## 전체 구조

이 모듈은 스파이크 정렬 과정의 다양한 단계에서 시각화를 제공합니다:
- 드리프트 보정 결과
- 진단 정보
- 스파이크 위치 분포

## 상수 정의

### `COLOR_CODES` (7줄)
기본 matplotlib 색상 코드 리스트입니다:
```python
COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
```
- 파란색, 초록색, 빨간색, 청록색, 자홍색, 노란색, 검은색, 흰색
- 여러 채널이나 클러스터를 구분하기 위해 순환 사용

### `PROBE_PLOT_COLORS` (9-20줄)
프로브 플롯용 색상 팔레트입니다:
- matplotlib의 tab10 색상 팔레트 기반
- 각 색상에 0.5 알파 채널 적용 (반투명)
- 마지막 색상은 회색 (비리프랙터리 유닛용)
- 총 10개 색상 (0-8: 리프랙터리 유닛, 9: 비리프랙터리 유닛)

## 주요 함수들

### 1. `plot_drift_amount()` (23-44줄)

드리프트 보정량을 시간에 따라 시각화합니다.

**입력 파라미터:**
- `ops`: 설정 및 결과 딕셔너리
  - `ops['dshift']`: 드리프트 보정량 배열 (shape: [n_batches, n_sections])
  - `ops['settings']`: 설정 딕셔너리
- `results_dir`: 결과 저장 디렉토리

**구현 세부사항:**
1. **스타일 설정**: 다크 배경 사용 (`dark_background`)
2. **시간 축 계산**: 
   ```python
   t = np.arange(dshift.shape[0])*(NT/fs)
   ```
   - 각 배치의 시간 위치 계산
   - `NT`: 배치 크기, `fs`: 샘플링 레이트
3. **다중 섹션 플롯**: 
   - 프로브의 각 섹션별로 다른 색상으로 플롯
   - `COLOR_CODES`를 순환 사용
4. **저장**: `drift_amount.png`로 저장 (300 DPI)

**출력:**
- 시간(x축) vs 깊이 보정량(y축) 그래프
- 각 프로브 섹션별로 다른 색상의 선

### 2. `plot_drift_scatter()` (47-75줄)

스파이크의 시간, 깊이, 진폭을 산점도로 시각화합니다.

**입력 파라미터:**
- `st0`: 스파이크 정보 배열 (shape: [n_spikes, 6])
  - `st0[:,0]`: 스파이크 시간 (초)
  - `st0[:,1]`: 스파이크 중심 깊이 (마이크론)
  - `st0[:,2]`: 스파이크 진폭
- `results_dir`: 결과 저장 디렉토리

**구현 세부사항:**
1. **진폭 클리핑**:
   ```python
   z[z < 10] = 10
   z[z > 100] = 100
   ```
   - 진폭을 10-100 범위로 제한하여 색상 매핑 안정화

2. **색상 매핑**:
   - 로그 스케일로 진폭을 90개 구간으로 분할
   - `matplotlib.colormaps['binary']` 사용
   - 각 구간의 평균 진폭을 색상으로 매핑
   - 진폭이 클수록 밝은 색상

3. **대형 플롯**: 30x14 인치 크기로 생성 (고해상도)

**출력:**
- 시간(x축) vs 깊이(y축) 산점도
- 색상 강도는 진폭에 비례
- `drift_scatter.png`로 저장

**용도:**
- 드리프트 보정의 효과 확인
- 스파이크 분포의 시간적 변화 관찰

### 3. `plot_diagnostics()` (78-121줄)

정렬 과정의 진단 정보를 4개의 서브플롯으로 표시합니다.

**입력 파라미터:**
- `Wall0`: 초기 클러스터 템플릿 (torch.Tensor, shape: [n_clusters, n_channels, n_pcs])
- `clu0`: 초기 클러스터 ID 배열 (np.ndarray)
- `ops`: 설정 및 결과 딕셔너리
  - `ops['wPCA']`: 시간적 주성분 (torch.Tensor)
  - `ops['settings']`: 설정 딕셔너리
- `results_dir`: 결과 저장 디렉토리

**4개의 서브플롯:**

#### Top Left: Temporal Features (84-90줄)
- **내용**: 시간적 주성분 (wPCA) 플롯
- **구현**:
  ```python
  t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
  ```
  - 시간 축을 밀리초 단위로 변환
  - 각 주성분을 다른 색상으로 플롯
- **의미**: 스파이크 웨이브폼의 시간적 특징

#### Top Right: Spatial Features (92-97줄)
- **내용**: 공간적 특징 히트맵
- **구현**:
  ```python
  features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
  axes[0][1].imshow(features.T, aspect='auto', vmin=0, vmax=25, cmap='binary_r')
  ```
  - 템플릿의 L2 노름 계산 (PC 차원 제거)
  - 채널(x축) vs 유닛(y축) 히트맵
- **의미**: 각 클러스터가 어떤 채널에서 활성화되는지

#### Bottom Left: Unit Amplitudes (105-110줄)
- **내용**: 유닛별 평균 진폭
- **구현**:
  ```python
  mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()
  ```
  - 각 클러스터 템플릿의 전체 L2 노름
- **의미**: 클러스터별 신호 강도

#### Bottom Right: Amplitude vs Spike Count (111-115줄)
- **내용**: 진폭 vs 스파이크 수 산점도
- **구현**:
  ```python
  spike_counts = np.bincount(clu0, minlength=n_units)  # 최적화됨
  axes[1][1].scatter(np.log(1 + spike_counts), mean_amp, s=3)
  ```
  - 각 클러스터의 스파이크 수 계산 (최적화: O(n) 복잡도)
  - 로그 스케일로 변환하여 플롯
- **의미**: 클러스터의 활성도와 신호 강도 관계

**최적화:**
- 이전에는 O(n_units × n_spikes) 복잡도의 루프 사용
- 현재는 `np.bincount`로 O(n_spikes)로 최적화

**출력:**
- 16x16 인치 크기의 2x2 서브플롯
- `diagnostics.png`로 저장

### 4. `plot_spike_positions()` (124-154줄)

스파이크 위치를 프로브 상에서 시각화합니다.

**입력 파라미터:**
- `clu`: 클러스터 ID 배열 (중복 제거 후)
- `is_refractory`: 리프랙터리 기간 만족 여부 (boolean 배열)
- `results_dir`: 결과 저장 디렉토리
  - `spike_positions.npy` 파일이 있어야 함

**구현 세부사항:**
1. **클러스터 색상 매핑**:
   ```python
   clu = clu.copy()
   bad_units = np.unique(clu)[is_refractory == 0]
   bad_idx = np.in1d(clu, bad_units)
   clu = np.mod(clu, 9)  # 0-8로 모듈로 연산
   clu[bad_idx] = 9      # 비리프랙터리 유닛은 9 (회색)
   ```
   - 리프랙터리 유닛: 0-8 색상 (9개 색상 순환)
   - 비리프랙터리 유닛: 9 (회색, 반투명도 0.25)

2. **위치 데이터 로드**:
   ```python
   positions = np.load(results_dir / 'spike_positions.npy')
   xs, ys = positions[:,0], positions[:,1]
   ```
   - `spike_positions.npy`에서 위치 정보 로드
   - x: 횡방향 위치, y: 깊이

3. **플롯**:
   ```python
   ax.scatter(ys, xs, s=3, c=colors)
   ```
   - 깊이(y축) vs 횡방향(x축) 산점도
   - 각 스파이크는 해당 클러스터 색상으로 표시

**출력:**
- 30x14 인치 대형 플롯
- 프로브 상의 스파이크 위치 분포
- 클러스터별 색상 구분
- `spike_positions.png`로 저장

**용도:**
- 클러스터의 공간적 분포 확인
- 리프랙터리 기간을 만족하는 좋은 유닛 식별
- 프로브 상의 스파이크 밀도 관찰

## 공통 패턴

모든 플롯 함수는 다음 패턴을 따릅니다:

1. **스타일 설정**: `plt.style.use('dark_background')`
2. **플롯 생성**: matplotlib figure/axes 생성
3. **데이터 처리**: PyTorch 텐서를 numpy로 변환
4. **시각화**: matplotlib 함수로 플롯
5. **레이블 및 제목**: 축 레이블, 제목 설정
6. **레이아웃**: `fig.tight_layout()`로 여백 조정
7. **저장**: 300 DPI로 PNG 파일 저장
8. **정리**: 스타일 리셋 및 figure 닫기

## 성능 고려사항

### 최적화된 부분:
- `plot_diagnostics()`의 스파이크 카운트 계산: `np.bincount` 사용
  - 이전: O(n_units × n_spikes) 루프
  - 현재: O(n_spikes) 벡터화 연산

### 메모리 고려사항:
- 대용량 데이터셋의 경우 `plot_drift_scatter()`가 메모리 집약적일 수 있음
- 모든 스파이크를 한 번에 플롯하므로 스파이크 수가 많으면 느려질 수 있음

### GPU/CPU 전환:
- PyTorch 텐서는 `.cpu().numpy()`로 변환하여 matplotlib에서 사용
- GPU 메모리에서 CPU 메모리로 데이터 이동

## 사용 예시

```python
from pathlib import Path
from kilosort import plots as kplots

results_dir = Path('path/to/results')

# 드리프트 플롯
kplots.plot_drift_amount(ops, results_dir)
kplots.plot_drift_scatter(st0, results_dir)

# 진단 플롯
kplots.plot_diagnostics(Wall0, clu0, ops, results_dir)

# 스파이크 위치 플롯
kplots.plot_spike_positions(clu, is_refractory, results_dir)
```

## 관련 파일

- `kilosort/run_kilosort.py`: 이 플롯 함수들을 호출하는 메인 파이프라인
- `kilosort/gui/sanity_plots.py`: GUI용 플롯 함수들 (PyQtGraph 사용)
- `kilosort/io.py`: `spike_positions.npy` 파일 생성

## 참고사항

- 모든 플롯은 다크 배경 스타일을 사용하지만, 저장 후 기본 스타일로 복원
- 플롯 파일은 모두 300 DPI로 저장되어 고해상도 출력 가능
- GUI 모드에서는 이 함수들이 아닌 `gui/sanity_plots.py`의 함수들이 사용됨

## 비효율적인 부분 및 개선 방안

### 1. `plot_drift_scatter()` - 색상 매핑 루프 (59-63줄)

**문제점:**
```python
for i in np.unique(bin_idx):
    subset = (bin_idx == i)
    a = z[subset].mean()
    colors[subset] = cm(((a-10)/90))
```

**왜 비효율적인가:**
- 각 구간(bin)마다 전체 배열을 스캔하여 subset을 찾음: O(n_bins × n_spikes)
- 구간이 많을수록 (최대 90개) 반복 횟수가 증가
- 각 반복마다 boolean 배열 생성 및 평균 계산으로 메모리 할당 발생

**개선 방법:**
```python
# 벡터화된 버전
unique_bins = np.unique(bin_idx)
bin_means = np.array([z[bin_idx == i].mean() for i in unique_bins])
bin_colors = cm((bin_means - 10) / 90)
# 인덱싱으로 색상 할당
for i, bin_val in enumerate(unique_bins):
    colors[bin_idx == bin_val] = bin_colors[i]
```

또는 더 효율적으로:
```python
# pandas groupby 스타일의 벡터화
unique_bins, inverse_idx = np.unique(bin_idx, return_inverse=True)
bin_means = np.bincount(inverse_idx, weights=z) / np.bincount(inverse_idx)
bin_colors = cm((bin_means - 10) / 90)
colors = bin_colors[inverse_idx]
```

**예상 성능 향상:**
- 구간이 90개, 스파이크가 100만 개일 때: 약 10-50배 빠름
- 메모리 사용량도 감소

### 2. `plot_spike_positions()` - 색상 매핑 루프 (137-140줄)

**문제점:**
```python
for i in range(10):
    subset = (clu == i)
    rgba = PROBE_PLOT_COLORS[i]
    colors[subset] = rgba
```

**왜 비효율적인가:**
- 각 색상(10개)마다 전체 배열을 스캔하여 subset을 찾음: O(n_colors × n_spikes)
- 스파이크 수가 많을수록 (수백만 개) 반복 횟수와 메모리 접근 증가
- 각 반복마다 boolean 배열 생성으로 메모리 할당 발생

**개선 방법:**
```python
# 인덱싱으로 직접 색상 할당 (벡터화)
colors = PROBE_PLOT_COLORS[clu]
```

**예상 성능 향상:**
- 스파이크가 100만 개일 때: 약 10-20배 빠름
- 코드도 더 간결하고 읽기 쉬움
- 메모리 사용량 감소

### 3. `plot_diagnostics()` - wPCA 플롯 루프 (86-88줄)

**문제점:**
```python
for i in range(wPCA.shape[0]):
    color = COLOR_CODES[i % len(COLOR_CODES)]
    axes[0][0].plot(t, wPCA[i,:].cpu().numpy(), c=color)
```

**왜 비효율적인가:**
- 루프 안에서 매번 `.cpu().numpy()` 호출로 GPU→CPU 전환 발생
- 각 주성분마다 개별 `plot()` 호출로 matplotlib 오버헤드 증가
- 주성분이 많을수록 (보통 3-6개) 반복 횟수 증가

**개선 방법:**
```python
# 미리 CPU로 변환
wPCA_np = wPCA.cpu().numpy()
# 한 번에 플롯 (또는 최소한 변환은 한 번만)
for i in range(wPCA_np.shape[0]):
    color = COLOR_CODES[i % len(COLOR_CODES)]
    axes[0][0].plot(t, wPCA_np[i,:], c=color)
```

또는 더 효율적으로 한 번에 플롯:
```python
wPCA_np = wPCA.cpu().numpy()
axes[0][0].plot(t, wPCA_np.T)  # 모든 주성분을 한 번에
# 색상은 자동으로 할당되거나 별도로 설정
```

**예상 성능 향상:**
- GPU→CPU 전환 오버헤드 제거: 약 2-5배 빠름
- matplotlib 호출 횟수 감소로 오버헤드 감소

### 4. `plot_diagnostics()` - 이미 최적화된 부분 (102줄)

**이미 개선됨:**
```python
# 이전 버전 (비효율적)
for i in range(n_units):
    spike_counts[i] = (clu0[clu0 == i]).size  # O(n_units × n_spikes)

# 현재 버전 (최적화됨)
spike_counts = np.bincount(clu0, minlength=n_units)  # O(n_spikes)
```

**왜 개선되었는가:**
- 이전: 각 클러스터마다 전체 배열을 스캔하여 O(n_units × n_spikes) 복잡도
- 현재: 한 번의 패스로 카운트하여 O(n_spikes) 복잡도
- 클러스터가 많을수록 (수천 개) 성능 차이가 극대화됨

**성능 향상:**
- 클러스터 1000개, 스파이크 100만 개일 때: 약 1000배 빠름

### 5. 전체적인 최적화 권장사항

**메모리 효율성:**
- 대용량 데이터셋의 경우 샘플링 고려:
  ```python
  if len(st0) > 1e6:
      sample_idx = np.random.choice(len(st0), int(1e6), replace=False)
      st0_sampled = st0[sample_idx]
      # 샘플링된 데이터로 플롯
  ```

**GPU 메모리 관리:**
- 모든 텐서를 한 번에 CPU로 변환하여 GPU 메모리 해제
- 큰 텐서는 청크 단위로 처리

**벡터화 원칙:**
- Python 루프 대신 NumPy 벡터화 연산 사용
- Boolean 인덱싱보다는 직접 인덱싱 사용
- 반복적인 배열 스캔 최소화

**예상 전체 성능 향상:**
- 위의 모든 최적화를 적용하면 전체 플롯 생성 시간이 **2-10배** 단축 가능
- 특히 대용량 데이터셋에서 효과가 큼
