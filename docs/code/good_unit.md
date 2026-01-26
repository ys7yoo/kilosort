# Good unit 판정 기준 (Kilosort)

Kilosort에서 **good unit(= single unit로 볼만한 클러스터)** 여부는 주로 **ACG(autocorrelogram)의 refractory period violation(오염) 정도**로 판단하며, 결과 저장 단계에서 각 클러스터에 `good`/`mua` 레이블을 붙여 `Phy`가 읽을 수 있는 TSV로 저장합니다.

## 요약

- **good 판정 조건**: \(R12 < \texttt{acg\_threshold}\) AND \(Q12 < 0.2\)
- **기본값**: `acg_threshold = 0.2` (더 엄격하게 하려면 0.1 권장 주석 존재)
- **레이블 저장 파일**
  - `cluster_KSLabel.tsv`: 클러스터별 `good`/`mua`
  - `cluster_group.tsv`: `cluster_KSLabel.tsv` 복사본(Phy 호환)
  - `cluster_ContamPct.tsv`: 추정 오염률(%) = `R12 * 100`

## 1) good 판정 로직

판정은 `kilosort/CCG.py`의 `check_CCG()`에서 수행됩니다.

- **good(= is_refractory) 조건**:
  - `R12 < acg_threshold`
  - `Q12 < 0.2`

참고: `cross_refractory`는 CCG 기반(클러스터 split/merge 등)에 사용되는 별도 조건이며, `good` 레이블 자체는 `is_refractory`를 사용합니다.

## 2) R12, Q12의 의미(코드 기준)

`kilosort/CCG.py`의 `CCG_metrics()`에서:

- **R12**: 짧은 지연 구간(= refractory 근처)에서의 이벤트율이 **baseline 대비 얼마나 큰지**를 나타내는 비율(작을수록 refractory가 잘 지켜짐)
- **Q12**: 위 감소가 우연이 아닐 가능성을 보는 통계적 지표(코드에서는 `Qi`의 최소값을 사용)

## 3) 레이블이 만들어지는 시점과 출력 파일

`kilosort/io.py`의 `save_to_phy()`에서 refractory 판정을 돌린 뒤, 각 클러스터에 레이블을 씁니다.

- `is_ref == True`  → `good`
- `is_ref == False` → `mua`

그리고 `cluster_group.tsv`는 `cluster_KSLabel.tsv`를 그대로 복사해서 생성됩니다(Phy가 표준적으로 읽는 파일명).

## 4) 기본 파라미터 위치

기본 임계값은 `kilosort/parameters.py`에 정의되어 있으며 설명에도 “good unit 할당에 사용”이라고 명시돼 있습니다.

- `acg_threshold`: default `0.2`
- `ccg_threshold`: default `0.25` (주로 split/merge용)

## 5) 코드에서 good 클러스터를 쓰는 예

`kilosort/data_tools.py`의 `get_good_cluster()`는 `cluster_KSLabel.tsv`에서 `good` 레이블만 골라 클러스터 id를 반환합니다.

