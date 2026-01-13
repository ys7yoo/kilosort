# run_kilosort.py 코드 설명

이 파일은 Kilosort4 스파이크 정렬 파이프라인의 메인 실행 모듈입니다. 신경 스파이크 데이터를 처리하여 스파이크를 검출하고 클러스터링하는 전체 파이프라인을 제공합니다.

## 전체 구조

이 파일은 신경 스파이크 정렬 파이프라인을 실행하는 여러 함수들을 포함하고 있으며, 전처리부터 최종 클러스터링까지의 모든 단계를 조율합니다.

## 주요 함수들

### 1. `run_kilosort()` (34-196줄)

메인 진입점 함수로, 사용자가 호출하는 공개 API입니다.

**주요 기능:**
- 설정 딕셔너리와 프로브 정보를 받아 전체 파이프라인을 실행
- 여러 샹크(shank) 지원 (순차적으로 처리)
- `_sort()` 함수를 호출하여 실제 정렬 수행

**입력 파라미터:**
- `settings`: 필수 설정 딕셔너리 (최소한 `n_chan_bin` 필요)
- `probe`: 프로브 딕셔너리 또는 `probe_name`으로 프로브 파일 지정
- `filename`: 데이터 파일 경로 (단일 파일 또는 리스트)
- `data_dir`: 데이터 디렉토리
- `device`: PyTorch 디바이스 (GPU/CPU)
- 기타 옵션들 (CAR, 드리프트 보정, 캐시 관리 등)

**반환값:**
- `ops`: 설정 및 결과를 담은 딕셔너리
- `st`: 스파이크 시간 배열 (3열: 시간, 템플릿, 진폭)
- `clu`: 클러스터 ID 배열
- `tF`: PC features (스파이크별 특징)
- `Wall`: 템플릿 (클러스터별 웨이브폼)
- `similar_templates`: 클러스터 간 유사도 행렬
- `is_ref`: 리프랙터리 기간 만족 여부
- `est_contam_rate`: 추정 오염률
- `kept_spikes`: 중복 제거 후 남은 스파이크 마스크

### 2. `_sort()` (199-366줄)

실제 정렬 파이프라인을 실행하는 내부 함수입니다.

**실행 단계:**

1. **시스템 정보 로깅**
   - GPU/CPU 자동 감지 및 설정
   - 시스템 정보 기록

2. **초기화** (`initialize_ops()`)
   - 설정 딕셔너리와 프로브 정보를 `ops`로 통합
   - 파라미터 검증

3. **전처리** (`compute_preprocessing()`)
   - 고역 통과 필터 생성
   - 화이트닝 행렬 계산
   - `BinaryFiltered` 객체 생성

4. **드리프트 보정** (`compute_drift_correction()`)
   - `datashift.run()` 호출
   - 보정된 `BinaryFiltered` 객체 반환
   - 드리프트 플롯 생성

5. **스파이크 검출** (`detect_spikes()`)
   - 템플릿 기반 초기 검출
   - 첫 번째 클러스터링
   - 학습된 웨이브폼으로 재검출

6. **클러스터링** (`cluster_spikes()`)
   - 그래프 기반 클러스터링
   - 클러스터 병합
   - 리프랙터리 기간 검사

7. **결과 저장** (`save_sorting()`)
   - Phy 포맷으로 저장
   - 통계 정보 기록

**에러 처리:**
- CUDA 메모리 부족 시 상세 정보 로깅
- 모든 예외를 로그 파일에 기록

### 3. `set_files()` (369-441줄)

파일 경로와 프로브 설정을 처리하는 함수입니다.

**주요 기능:**
- 데이터 파일 찾기 (`io.find_binary()` 사용)
- 결과 디렉토리 생성 (기본값: `data_dir / 'kilosort4'`)
- 프로브 파일 로드
- 배드 채널 제거
- 샹크 선택 (특정 샹크만 처리)

**경로 처리:**
- `filename`이 제공되면 해당 파일 사용
- `data_dir`만 제공되면 디렉토리에서 바이너리 파일 자동 검색
- 여러 파일을 리스트로 받아 시간 순서로 연결 처리

### 4. `setup_logger()` / `close_logger()` (444-482줄)

로깅 시스템을 설정하는 함수들입니다.

**`setup_logger()`:**
- 파일 핸들러: `kilosort4.log`에 DEBUG 레벨로 저장
- 콘솔 핸들러: INFO 레벨 (verbose 옵션 시 DEBUG)
- 타임스탬프 및 로그 레벨 포함

**`close_logger()`:**
- 모든 핸들러 제거 및 정리

### 5. `initialize_ops()` (485-553줄)

설정과 프로브 정보를 `ops` 딕셔너리로 통합합니다.

**주요 작업:**
- 기본 설정값 적용 (`DEFAULT_SETTINGS`와 병합)
- 파라미터 검증 (인식되지 않은 설정 경고)
- `ops` 딕셔너리 구성:
  - 배치 크기 계산 (`NTbuff = batch_size + 2 * nt`)
  - 채널 수 설정
  - 중복 스파이크 검사 간격 계산
  - 프로브 정보 통합

**검증:**
- `nearest_chans`가 채널 수를 초과하지 않도록 조정
- `templates_from_data=False`일 때 `nt=61` 필수
- 프로브 배열 차원 검사

### 6. `compute_preprocessing()` (581-656줄)

전처리 파라미터를 계산하고 저장합니다.

**주요 작업:**
1. **고역 통과 필터 생성**
   - `preprocessing.get_highpass_filter()` 호출
   - 샘플링 레이트와 컷오프 주파수 기반

2. **화이트닝 행렬 계산**
   - `preprocessing.get_whitening_matrix()` 호출
   - 채널 간 상관관계 제거

3. **BinaryFiltered 객체 생성**
   - 필터링된 데이터 접근을 위한 래퍼
   - 배치 단위로 데이터 처리

**저장 정보:**
- `ops['preprocessing']['whiten_mat']`: 화이트닝 행렬
- `ops['preprocessing']['hp_filter']`: 고역 통과 필터
- `ops['Wrot']`, `ops['fwav']`: 별칭
- 런타임 및 리소스 사용량 기록

### 7. `compute_drift_correction()` (659-743줄)

드리프트 보정을 수행합니다.

**주요 작업:**
1. **드리프트 보정 실행**
   - `datashift.run()` 호출
   - 블록 단위로 드리프트 추정 및 보정

2. **보정된 BinaryFiltered 생성**
   - 드리프트 보정이 적용된 데이터 접근 객체
   - 후속 단계에서 사용

**출력:**
- `ops['dshift']`: 드리프트 보정량
- `ops['yblk']`: 블록 위치
- `st0`: 중간 스파이크 시간 (플롯용)
- 런타임 및 CUDA 메모리 통계

### 8. `detect_spikes()` (746-865줄)

스파이크를 검출하는 함수입니다.

**3단계 프로세스:**

1. **템플릿 기반 초기 검출**
   - `spikedetect.run()` 호출
   - 범용 템플릿 또는 데이터에서 학습한 템플릿 사용
   - PC features (`tF`) 추출

2. **첫 번째 클러스터링**
   - `clustering_qr.run()` 호출 (mode='spikes')
   - 초기 클러스터 할당
   - 템플릿 후처리 (`template_matching.postprocess_templates()`)

3. **학습된 웨이브폼으로 재검출**
   - `template_matching.extract()` 호출
   - 클러스터별 학습된 템플릿 사용
   - 더 정확한 스파이크 검출

**출력:**
- `st`: 최종 스파이크 시간 배열 (3열)
- `tF`: PC features
- `Wall`: 클러스터 템플릿
- `clu`: 초기 클러스터 ID

### 9. `cluster_spikes()` (868-951줄)

최종 클러스터링을 수행합니다.

**주요 단계:**

1. **그래프 기반 클러스터링**
   - `clustering_qr.run()` 호출 (mode='template')
   - PC features 기반 클러스터링
   - QR 분해를 이용한 효율적 계산

2. **클러스터 병합**
   - `template_matching.merging_function()` 호출
   - 유사한 클러스터 병합
   - 리프랙터리 기간 검사 (`check_dt=True`)

**출력:**
- `clu`: 최종 클러스터 ID
- `Wall`: 병합된 클러스터 템플릿
- `is_ref`: 리프랙터리 기간 만족 여부
- `st`, `tF`: 병합 후 업데이트된 스파이크 정보

### 10. `save_sorting()` (954-1063줄)

정렬 결과를 저장하고 Phy 포맷으로 내보냅니다.

**주요 작업:**

1. **Phy 포맷 저장**
   - `io.save_to_phy()` 호출
   - Phy에서 사용 가능한 모든 파일 생성

2. **리프랙터리 기간 계산**
   - ACG (AutoCorrelogram) 분석
   - CCG (CrossCorrelogram) 분석
   - 오염률 추정

3. **통계 정보 기록**
   - 총 유닛 수, 좋은 유닛 수
   - 스파이크 수
   - 평균 드리프트량
   - 총 런타임

4. **ops 저장**
   - `ops.npy` 파일로 저장
   - 모든 설정 및 결과 포함

**저장 파일:**
- `spike_times.npy`: 스파이크 시간
- `spike_clusters.npy`: 클러스터 ID
- `similar_templates.npy`: 클러스터 유사도
- `kept_spikes.npy`: 중복 제거 마스크
- `params.py`: Phy 설정 파일
- 기타 Phy 포맷 파일들

### 11. `load_sorting()` (1066-1163줄)

저장된 정렬 결과를 로드하는 함수입니다.

**기본 로드:**
- `ops.npy`: 설정 및 결과
- `spike_times.npy`: 스파이크 시간
- `spike_clusters.npy`: 클러스터 ID
- `similar_templates.npy`: 클러스터 유사도
- `kept_spikes.npy`: 중복 제거 마스크
- 리프랙터리 기간 재계산

**추가 변수 로드** (`load_extra_vars=True`):
- `tF.npy`: PC features
- `Wall.npy`: 클러스터 템플릿
- `full_st.npy`: 중복 제거 전 전체 스파이크
- `full_clu.npy`: 중복 제거 전 전체 클러스터 ID
- `full_amp.npy`: 전체 진폭 정보

## 보조 함수

### `get_run_parameters()` (555-578줄)

`ops` 딕셔너리에서 자주 사용되는 파라미터들을 추출하여 리스트로 반환합니다. 여러 함수에서 공통으로 사용되는 파라미터를 일관되게 전달하기 위한 헬퍼 함수입니다.

## 주요 특징

### 1. GPU/CPU 지원
- PyTorch 기반 계산
- CUDA 자동 감지 및 설정
- CPU 폴백 지원

### 2. 메모리 관리
- `clear_cache` 옵션으로 GPU 메모리 관리
- 대용량 데이터 처리 최적화

### 3. 진행 상황 추적
- `progress_bar` 파라미터로 GUI/CLI 진행 상황 표시
- tqdm 또는 Qt ProgressBar 지원

### 4. 상세한 로깅
- 모든 단계의 로그 파일 생성
- 성능 메트릭 기록
- CUDA 메모리 사용량 추적

### 5. 다중 파일 지원
- 여러 바이너리 파일을 시간 순서로 연결
- 긴 기록을 여러 파일로 나눈 경우 처리

### 6. 샹크별 처리
- 여러 샹크를 순차적으로 처리
- 각 샹크별로 별도 결과 디렉토리 생성

### 7. 에러 처리
- CUDA 메모리 부족 시 상세 정보 제공
- 모든 예외를 로그에 기록
- 사용자 친화적 에러 메시지

## 사용 예시

```python
from kilosort import run_kilosort

# 기본 사용
settings = {'n_chan_bin': 385, 'fs': 30000}
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings, probe_name='neuropixels1', filename='data.bin')

# GPU 명시적 지정
import torch
device = torch.device('cuda:0')
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings, probe_name='neuropixels1', filename='data.bin', device=device)

# 여러 파일 연결
filenames = ['data_part1.bin', 'data_part2.bin', 'data_part3.bin']
ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings, probe_name='neuropixels1', filename=filenames)
```

## 관련 모듈

- `kilosort.preprocessing`: 전처리 (필터링, 화이트닝)
- `kilosort.datashift`: 드리프트 보정
- `kilosort.spikedetect`: 스파이크 검출
- `kilosort.template_matching`: 템플릿 매칭 및 병합
- `kilosort.clustering_qr`: QR 분해 기반 클러스터링
- `kilosort.io`: 파일 입출력
- `kilosort.CCG`: 리프랙터리 기간 분석
- `kilosort.parameters`: 기본 설정값
