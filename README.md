해당 프로젝트는 보안 사항으로 인해 구체적인 결과물 및 코드는 비공개이며, 수행한 업무 중심으로만 내용을 작성하였습니다.

# AI_Shortform_Generator

> 가로 영상을 세로 비율로 변환하고, 주요 인물을 중심에 배치하는 자동화 시스템


# 1. Role  
- Object Detection  
- Object Tracking  


# 2. Object Detection

## 2.1 Detection의 필요성

- 가로 비율의 영상을 세로로 변환하는 과정에서 **모든 인물을 프레임 내에 담는 것은 불가능**
- 장면마다 중심이 되는 **주요 인물**을 기준으로 리프레임이 필요함
- 따라서, **객체 탐지를 통해 주요 인물을 식별**하고, 이를 기준으로 중심을 재설계하는 과정이 필요


## 2.2 Head + Person 모델 조합

### 2.2.1 전략

- 영상 중심에 나타나는 객체의 <b>Head(머리)<b>가 가장 중요한 정보라고 판단하여 **Head Detection 모델**을 먼저 설계
- 다만, 얼굴이 보이지 않는 경우(하반신만 보이는 인물 등)에 대한 대응을 위해 **Person Detection 모델**을 병행
- Head 중심만으로 영상 중심을 잡으면 위쪽으로 쏠리고, Person 중심만으로 잡으면 비자연스러운 장면이 발생
- 따라서 **Head와 Person 중심을 적절한 비율로 혼합하여 자연스러운 중심점**을 도출

### 2.2.2 파인튜닝 및 결과

- 사용 모델: **RT-DETR**
  - [논문 링크](https://arxiv.org/pdf/2411.18164)
  - 성능 및 상용화 가능성 고려

- 학습 데이터셋:
  - Head: [person-head-face-v4](https://universe.roboflow.com/dohien-rzu9v/person-head-face-v4)
  - Person: [Crowd Detection](https://universe.roboflow.com/ricky-sambora/crowd-detection-7suou)

- 성능 결과:

  | Metric        | Head     | Person   |
  |---------------|----------|----------|
  | Precision     | 0.7611   | 0.8965   |
  | Recall        | 0.6522   | 0.9166   |
  | mAP@0.5       | 0.7246   | 0.9403   |
  | mAP@0.5:0.95  | 0.4197   | 0.7600   |

- 문제점:
  - 실제 테스트 영상에서는 FP(False Positive)가 빈번하게 발생
  - Confidence score 분포가 넓어 threshold 조정이 어렵고, 탐지 결과의 신뢰도가 낮았음


## 2.3 Bidirectional Loss-Guided Learning Rate Scheduler 제안

> **아이디어 요약**  
> 학습 중 loss가 개선되지 않을 경우, 학습률을 **증가와 감소 방향으로 각각 분기하여 병렬로 시도**한 뒤,  
> **loss가 다시 감소한 쪽의 학습률을 채택**하고 이 과정을 반복하는 방식의 스케줄러를 제안함.

### 제안 배경

- 일반적으로 학습률 감소는 **loss의 최적점 수렴을 위한 수단**,  
  학습률 증가는 **local minima 탈출을 위한 전략**으로 활용됨.
- 현재 사용되는 스케줄러는 대부분 일정 주기로 학습률을 증가/감소시키거나,  
  `ReduceLROnPlateau`처럼 **loss가 일정 기간 개선되지 않을 경우 감소만 적용**함.
- 하지만 **loss 추이를 기반으로 학습률을 양방향(증가/감소) 모두 고려하는 스케줄러는 부재**함.

### 작동 방식

1. 일정 step 동안 loss 개선이 없을 경우,  
   현재 학습률을 기준으로 **증가 방향과 감소 방향으로 각각 분기하여 병렬 학습 진행**
2. 양쪽 결과 중 **loss가 감소한 쪽의 학습률을 채택**, 나머지는 폐기
3. 위 과정을 **반복**

### 한계점

- 시간이 부족하여 실제 구현 및 검증은 진행하지 못하고 **설계 단계에서 중단**

## 2.4 Face + Person 모델 조합

- Head Detection 모델의 한계를 보완하기 위해 **Face Detection 모델을 대체 적용**
- 사용 모델:
  - Face: **BAFALLO** (pretrained)
  - Person: **RT-DETR** (pretrained, fine-tuning 없이 사용)

- 테스트 영상 결과:
  - Face는 매우 높은 정확도를 보였으며, 신뢰도가 높음
  - Person은 객체를 대부분 탐지했으나, person이 아닌 객체도 인식하거나 하나의 객체에 대해 중복 bbox가 생성되는 문제가 있었음


# 3. Object Tracking

## 3.1 Deep SORT 오픈소스 활용

- 객체가 다른 객체에 가려지거나 재등장하는 경우, bbox가 발산하는 현상이 보임
- 특히, 다수 인물이 군집된 상황에서 **ID 충돌, 누락, 재할당 오류**가 다수 발생

## 3.2 커스텀 Tracking 시스템

### 3.2.1 Face Tracking

- 탐지 신뢰도가 높은 Face를 기준 객체로 설정
- **칼만 필터 기반 ID 부여**, **선형 보간**, **이동 평균 스무딩** 기법을 활용해 안정적 추적 구현

### 3.2.2 Face-Person Matching

- 매칭 조건:
  1. Person bbox 내부에 Face bbox가 포함되어야 함
  2. Face의 면적이 Person bbox 대비 특정 비율 내에 있어야 함
- 각 Face 객체에 대해 가장 가까운 Person을 1:1로 매칭

### 3.2.3 Matched Person Tracking

- 매칭된 Person 객체에 대해 **보간 + 이동 평균 + 중심 제어 기반의 스무딩** 적용

### 3.2.4 Unmatched Person 처리

- Face가 탐지되지 않은 Person 객체 중:
  - Confidence가 높은 것만 사용
  - 기존에 매칭된 bbox와 겹치는 객체는 제거하여 중복 제거

### 3.2.5 Unmatched Person Tracking

- 칼만 필터 기반 ID 부여
- 선형 보간 및 중심 제어로 추적 안정성 확보

### 3.2.6 중심 병합 전략

- Face 중심과 Person 중심을 적절한 비율로 혼합하여 최종 중심점 생성
- 생성된 중심 기반으로:
  - **Face + Person 혼합 중심 데이터셋**
  - **Person-only 중심 데이터셋** 구성

> 이후 후처리 및 영상 생성 로직은 보안 사유로 비공개

