# V1 DEMO RUNBOOK

## 1. Set AWS credentials (same PowerShell window)

```powershell
$Env:AWS_DEFAULT_REGION="us-west-2"
$Env:AWS_ACCESS_KEY_ID="..."
$Env:AWS_SECRET_ACCESS_KEY="..."
$Env:AWS_SESSION_TOKEN="..."
```

## 2. Confirm the endpoint is InService

```powershell
.venv\Scripts\python.exe -c "import boto3;print(boto3.client('sagemaker',region_name='us-west-2').describe_endpoint(EndpointName='youbike-v1-multimodel-endpoint')['EndpointStatus'])"
```

## 3. Launch

```powershell
.venv\Scripts\streamlit run app.py
```

If the endpoint is unavailable the app shows
`V1 SageMaker Endpoint unavailable，已切換至 Stable V0。` and renders the
stable V0 UI instead. It never crashes and never falls back silently.

## Demo scenario data

- weekday snapshot: 2026-05-28 07:30:36 (1159 stations, is_peak=1 morning peak)
- weekend snapshot: 2026-05-31 19:30:32 (1159 stations)

The V1 models need `rainfall` and `is_peak`, which the V0 live snapshot
does not carry, so the demo uses a real historical scenario. The UI always
prints `Demo 情境時間`. Never described as live data.

## 還車情境 (Return)

1. 情境日型：平日（尖峰）
2. 使用者模式 → 選「我要還車」
3. 目的地輸入：`三井Outlet`
4. 搜尋範圍：500 m

預期結果（10 站在範圍內）：

- 最近可行站：**三井Outlet**（標示「最近」，獎勵金 —）
- 有回饋金的站：**新北市林口行政園區**
  - 風險狀態 `persistent_low`
  - 額外繞行 294.8 m
  - 獎勵金 **+$10**
  - 共 1 站顯示回饋金

## 借車情境 (Borrow)

1. 情境日型：平日（尖峰）
2. 使用者模式 → 選「我要借車」
3. 目的地輸入：`捷運江子翠站(1號出口)`
4. 搜尋範圍：500 m

預期結果（15 站在範圍內）：

- 最近可行站：**捷運江子翠站(1號出口)**（標示「最近」，獎勵金 —）
- 有回饋金的站：**捷運江子翠站(6號出口)**
  - 風險狀態 `persistent_high`
  - 額外繞行 168.4 m
  - 獎勵金 **+$8**
  - 共 3 站顯示回饋金

## Operator centre

切到「營運中心」，應看到：

- 4 個 KPI：高風險站點數 / 持續空車風險站 / 持續滿車風險站 / 目前監控站點
- 站點風險列表，欄位：站點名稱、站點地址、30分鐘滿車風險、1小時滿車風險、
  30分鐘空車風險、1小時空車風險
- 依風險排序（持續性風險在最前）
- 篩選：行政區 / 風險程度 / 站點搜尋

## Cost

Endpoint 持續計費（ml.m5.large，約 $0.115/hr）。Demo 全部結束後執行：

```powershell
v1_training\venv\Scripts\python.exe deploy_v1\delete_endpoint.py
```

## Known limitations

- `day_type` 由 calendar 判斷（Mon-Fri / Sat-Sun），**尚未**做國定假日 runtime 判斷。
- Demo 使用歷史情境 snapshot，非即時資料。
- 資料涵蓋新北市，`SOGO 忠孝館`（台北市）不在 coverage 內；輸入後會落到
  涵蓋範圍內的預設地點，不會產生假站點。