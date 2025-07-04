# 数据统计
- 数据量: 200
- 个体类型（每种类型20个）：
    - 一直正常
    - 一直异常
    - 一直预警
    - 正常预警
    - 预警到正常
    - 预警到异常
    - 正常预警异常
    - 异常预警正常
    - 正常预警正常
    - 异常预警异常

# 数据说明
## 1. user_profiles.csv
- **内容**：用户基本信息
- **字段**：
  - user_id：用户编号
  - age：年龄
  - gender：性别（男/女）
  - height_cm：身高（cm）
  - weight_kg：体重（kg）
  - BMI：体重指数
- **说明**：每行对应一名用户的基本属性。

## 2. user_groups.csv
- **内容**：用户分组信息
- **字段**：
  - user_id：用户编号
  - abnormal_group：异常分组（如心率异常组、血氧异常组、正常等）
  - individual_type：个体类型（如一直正常、一直异常、正常到预警等）
- **说明**：用于后续分组分析和异常检测。

## 3. medical_records.csv
- **内容**：用户电子病历
- **字段**：
  - user_id：用户编号
  - WBC, RBC, HGB, HCT, PLT, NEUT%, LYM%, GLU, CRP, ALT, AST, UA, CREA, BUN, TC, TG, HDL, LDL, SBP, DBP, TP, ALB, SpO2, Temp, BMI 等医学指标
- **说明**：每行对应一名用户的医学检查结果，包含血常规、代谢、血脂、血压、营养等多项指标。

## 4. daily_physiology.csv
- **内容**：用户一天的生理监测数据
- **字段**：
  - time：时间戳（精确到秒）
  - HR：心率
  - SpO2：血氧饱和度
  - step：每5秒步数
  - step_cum：累计步数
  - calories：热量消耗
  - activity：活动类型（如睡眠、活动、休闲等）
  - weather：天气
  - status：健康状态（0=正常，1=预警，2=异常）
- **说明**：每行为用户在某一时刻的生理与活动状态。

## 5. insole_data.csv
- **内容**：用户一天的足底压力与步态数据
- **字段**：
  - time：时间戳（精确到秒）
  - LF, LH, RF, RH：左前、左后、右前、右后足底压力
  - step_freq：步频
  - gait_type：步态类型（如正常、异常、急行、缓行、静息）
  - center_x, center_y：足底压力中心坐标
- **说明**：每行为用户在某一时刻的足底压力与步态特征。

## 6. medical_records_abnormal.csv
- **内容**：用户电子病历（含异常与预警状态）
- **字段**：
  - user_id：用户编号
  - 其他医学指标（同 medical_records.csv）
  - status：健康状态（0=正常，1=预警，2=异常）
- **说明**：
  - 该文件基于 user_groups.csv 的个体类型（individual_type）和异常分组（abnormal_group），对 medical_records.csv 进行扩展。
  - 当用户处于"预警"或"异常"状态时，相关医学指标会根据异常类型进行调整，且"预警"幅度为"异常"幅度的一半。
  - 例如：
    - 心率异常组：
      - 预警：CRP +1，WBC +0.75，NEUT% +4，LYM% -4
      - 异常：CRP +2，WBC +1.5，NEUT% +8，LYM% -8
    - 血氧异常组：
      - 预警：SpO2 -1，RBC +0.15，HCT +0.015
      - 异常：SpO2 -2，RBC +0.3，HCT +0.03
  - 每个用户可有多条记录，分别对应不同健康状态。

