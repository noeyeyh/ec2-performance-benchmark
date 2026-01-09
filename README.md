# EC2 인스턴스 기반 LLM 추론 성능 비교

## 개요

뉴스 전문을 요약해주는 동일한 작업에 대해 각 인스턴스 환경에서 LLM 작업을 여러 EC2 인스턴스에서 수행하여 **같은 모델의 메모리 크기별 비교**와 **같은 메모리(용량 급)에서 다른 인스턴스 모델 간 성능 비교**를 정리했습니다.

- **작업**: 2개의 기사 요약
- **결과 파일**: `summary.json`, `perf.json`
- **측정 지표**: 총 소요 시간(초), 프롬프트/출력 처리량(chars/s), 종료 시 RSS 메모리(MB), CPU 사용률(%)

> 참고: R5는 메모리 최적화형, M5는 범용(컴퓨트/메모리/네트워킹 균형), G5는 NVIDIA GPU 기반 추론/그래픽에 최적화된 계열입니다.

---

## 사용 데이터

본 프로젝트에서는 국내 네이버 뉴스 사이트를 대상으로 웹 크롤링을 수행하였습니다.

크롤링 과정에서는 뉴스 페이지의 HTML을 수집한 뒤, 본문 텍스트를 중심으로 파싱하고 광고,메뉴,네비게이션과 같은 불필요한 요소를 제거하여 순수 텍스트 데이터만을 추출하였습니다.
해당 크롤링 및 전처리 로직은 코드 레벨에서 모듈화하여 구현하였으며, 검색 키워드 기반으로 문서를 필터링할 수 있도록 구성했습니다.

이후 수집된 뉴스 데이터 중 검색 키워드 신한투자증권과 직접적으로 연관된 문서들을 선별하여 테스트용 데이터셋을 구성하였습니다.

이 데이터로 AWS 인스턴스 유형별 크롤링된 데이터 요약에 대한추론 성능을 측정하였습니다.

## 실험 목적

본 실험은 동일한 LLM 추론 작업을 기준으로,
EC2 인스턴스의 **크기와 인스턴스 유형 차이에 따라 추론 성능이 어떻게 달라지는지**를 비교하기 위해 수행되었습니다.

구체적으로는 다음과 같은 비교를 목표로 합니다.

- 같은 모델을 사용했을 때, **메모리 최적화형(R5) 인스턴스에서 인스턴스 크기(scale-up)에 따른 성능 변화**
- 같은 용량 급(16xlarge)에서, **서로 다른 EC2 인스턴스 모델(R5, M5, G5)을 사용하여 인스턴스 모델별 특화 사항에 따른 추론 성능 비교**

---

## 테스트 환경

- **OS**: Ubuntu (EC2)
- **모델**: `mistral:7b-instruct-q4_0`

---

## 1) 같은 모델 — R5 메모리 크기별 비교

`mistral:7b-instruct-q4_0`를 R5 계열에서 실행한 결과입니다.

| 인스턴스 타입   | vCPU/메모리(공식 스펙) | 총 소요 시간 (s) | 프롬프트 처리량 | 출력 처리량  | RSS 메모리 |
| --------------- | ---------------------- | ---------------- | --------------- | ------------ | ---------- |
| **r5.4xlarge**  | 16 vCPU / 128 GB       | 202.182          | 5.25 chars/s    | 3.91 chars/s | 12.38 MB   |
| **r5.8xlarge**  | 32 vCPU / 256 GB       | 69.578           | 15.26 chars/s   | 5.13 chars/s | 12.25 MB   |
| **r5.16xlarge** | 64 vCPU / 512 GB       | 49.829           | 21.31 chars/s   | 4.29 chars/s | 11.50 MB   |

<div align="center">

  <!-- 1행 (3개) -->
  <img style="width:32%;" src="https://github.com/user-attachments/assets/109637fc-bd34-42e4-b3c2-5b84e635efcc">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/3d8dd1d3-d9a8-4161-8964-0de02cadf3af">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/af50be37-11e1-4381-858d-fe94b3ab59ac">
  <br/>

  <!-- 2행 (3개) -->
  <img style="width:32%;" src="https://github.com/user-attachments/assets/29b4492b-a42e-4013-9129-ced145ecc8d0">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/a0df57bd-d325-41a4-9feb-366e646d1165">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/d8deac4f-0d8c-4105-ac1a-e585720548f6">

</div>

> R5 메모리/사이즈 스펙은 AWS 공식 문서에 근거합니다. (R5 [AWS](https://aws.amazon.com/ec2/instance-types/r5/))

**해석**

- 메모리가 커지고 vCPU가 늘어날수록 **총 소요 시간은 크게 감소**(r5.4xl → r5.8xl → r5.16xl).
- **프롬프트 처리량**도 선형에 가깝게 증가.
- **출력 처리량**은 8xlarge에서 가장 높고 16xlarge에서 다소 낮아 **토큰 생성(샘플링) 변동성** 가능성이 있음.
- **RSS 메모리**는 모두 ~12 MB 수준으로 유사 → 모델/스크립트 종료 시점의 RSS 기준이며 **피크 메모리**가 아님.

---

## 2) 같은 메모리(급) — 다른 인스턴스 모델 비교

동일한 모델을 다음 16xlarge 급(대략 64 vCPU)에서 비교했습니다.

| 인스턴스 타입   | 아키텍처/특화         | 총 소요 시간 (s) | 프롬프트 처리량 | 출력 처리량  | RSS 메모리 |
| --------------- | --------------------- | ---------------- | --------------- | ------------ | ---------- |
| **r5.16xlarge** | 메모리 최적화 (Nitro) | 49.829           | 21.31 chars/s   | 4.29 chars/s | 11.50 MB   |
| **g5.16xlarge** | NVIDIA A10G GPU 기반  | 41.351           | 25.68 chars/s   | 5.08 chars/s | 11.50 MB   |
| **m5.16xlarge** | 범용/AVX‑512 (Nitro)  | 37.753           | 28.13 chars/s   | 5.54 chars/s | 12.00 MB   |

<div align="center">

  <!-- 1행 (3개) -->
  <img style="width:32%;" src="https://github.com/user-attachments/assets/1e425a45-0fcb-4879-b77c-241fa090eb1a">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/dcd2e8c3-b587-4392-b0a8-2788411cbfda">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/0839e44a-2ac2-4bd6-b067-1cbcef9d54e2">
  <br/>

  <!-- 2행 (3개) -->
  <img style="width:32%;" src="https://github.com/user-attachments/assets/40c39788-32a3-4e68-8e29-403d76fa702f">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/ca844999-2bcc-473e-b7e4-68c86c7dfc2a">
  <img style="width:32%;" src="https://github.com/user-attachments/assets/fc494781-aa4d-40c9-8e69-fbbb2c7b36f3">

</div>

**해석**

- **m5.16xlarge**가 세 인스턴스 중 **가장 빠른 총 시간**과 **최고 프롬프트/출력 처리량**을 기록.
- **g5.16xlarge**는 GPU 기반임에도 해당 실행 조건에서 m5보다 느렸지만, R5보다는 빠름. (GPU 최적화 프레임워크·배치·정밀도 설정에 따라 결과가 크게 달라질 수 있음)
- **r5.16xlarge**는 메모리 최적화 계열로 대용량 인메모리 워크로드에 강점이 있으나, LLM 추론 속도는 CPU/GPU 계열 대비 낮게 측정.

---

## 3) 인스턴스 모델별 특화 사항 및 장단점 (공식 문서 기반)

### R5 (메모리 최적화형)

- **특화**: 대규모 인메모리 처리, 분석, 캐시/DB, 높은 메모리 대 vCPU 비율. Nitro 기반으로 호스트 리소스를 효율 제공.
- **장점**: 더 많은 메모리/대역폭, 인메모리 DB/분석에 최적. 다양한 변형(R5n/R5b/R5a). R5b는 **EBS 성능 최대 60 Gbps/260K IOPS** 제공.
- **출처**: R5 공식 페이지 [AWS](https://aws.amazon.com/ec2/instance-types/r5/)

### M5 (범용)

- **특화**: 균형 잡힌 컴퓨트/메모리/네트워크. Intel Xeon 기반, **AVX‑512/VNNI** 지원, Nitro 기반.
- **장점**: 다양한 워크로드(웹/앱 서버, 중형 DB, 캐시, 개발환경)에서 비용 대비 성능 우수. M5d/DN 변형은 로컬 NVMe 제공.
- **출처**: M5 공식 페이지 [AWS](https://aws.amazon.com/ec2/instance-types/m5/)

### G5 (GPU 기반)

- **특화**: NVIDIA **A10G** Tensor Core GPU, ML 추론/그래픽에서 **G4 대비 최대 3× 성능, 최대 40% 가격 대비 성능 개선**. 최대 192 vCPU, 100 Gbps 네트워크, 최대 7.6 TB NVMe.
- **장점**: 딥러닝 추론/훈련에서 높은 처리량과 낮은 지연, CUDA/TensorRT/cuDNN 생태계 활용.
- **출처**: G5 공식 페이지 [AWS](https://aws.amazon.com/ec2/instance-types/g5/), AWS ML 블로그 [링크](https://aws.amazon.com/blogs/machine-learning/achieve-four-times-higher-ml-inference-throughput-at-three-times-lower-cost-per-inference-with-amazon-ec2-g5-instances-for-nlp-and-cv-pytorch-models/)

---

## 4) 종합 결론

- 같은 모델(R5 계열 내)에서는 인스턴스 크기가 커질수록(메모리/CPU 증가) **전체 시간 단축, 프롬프트 처리량 증가** 경향.
- 같은 메모리 급(16xlarge 비교)에서는 **m5 > g5 > r5** 순으로 빠르게 측정됨. 이는 본 워크로드가 **CPU 연산/메모리 접근**에 상대적으로 민감하고, GPU 최적화(배치/정밀도/엔진)가 충분히 적용되지 않았기 때문일 수 있음.
- LLM 추론 전반에서는 **GPU 최적화가 이루어질수록** G5 계열이 비용 대비 성능에서 유리해지는 것이 일반적입니다. (공식 블로그 인사이트 참고)

  <img width="3569" height="2069" alt="set1_metrics" src="https://github.com/user-attachments/assets/f223a9d9-c631-4ec7-8e45-3049ffd2342d" />

<img width="3569" height="2069" alt="set2_metrics" src="https://github.com/user-attachments/assets/f393e2d3-b4ec-4227-87ea-6354d10e0e56" />

---

## 부록: 원시 측정값 요약

- r5.4xlarge — total **202.182s**, prompt **5.25**, output **3.91**, rss **12.38MB**
- r5.8xlarge — total **69.578s**, prompt **15.26**, output **5.13**, rss **12.25MB**
- r5.16xlarge — total **49.829s**, prompt **21.31**, output **4.29**, rss **11.50MB**
- g5.16xlarge — total **41.351s**, prompt **25.68**, output **5.08**, rss **11.50MB**
- m5.16xlarge — total **37.753s**, prompt **28.13**, output **5.54**, rss **12.00MB**
