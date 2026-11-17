# K：HMM 知識核心（k_core）

以「場域」為單位保存 AoV 演變序列、訓練 HMM 並進行推論。設計說明見 `../docs/設計文件.md`。

> 目前沒有真實工廠資料。示意資料都會標記 `is_synthetic=true`，其訓練結果只用來驗證程式與方法，不代表任何真實工廠成效。

## 安裝與啟動

需要 Python 3.11 以上版本。

```powershell
cd k_core/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn kcore.main:app --reload --port 8900
```

- API 測試介面：<http://localhost:8900/docs>
- 資料庫預設為 `k_core/backend/kcore.db`，可用環境變數 `KCORE_DATABASE_URL` 指定。
- 埠號 8900 是為了不和舊平台的 8800 衝突。

## 審核畫面（前端）

需要 Node.js 20 以上版本。請先依上一節啟動後端（8900 埠），再執行：

```powershell
cd k_core/frontend
npm install
npm run dev
```

用瀏覽器開啟 <http://localhost:5174>（網址加上 `#case/<案例代號>` 可直接開啟某個案例，例如 `#case/CVFLOW`），有四個分頁：

- **模型研究**（預設）：
  - **模型概況**：選擇模型版本，比較 HMM、兩種「HMM＋上一次修改」、只看上一次修改、只看頻率在測試資料上的困惑度與 Top-1／Top-3。
  - **隱藏狀態**：每個狀態最可能產生的觀測與實際修改組成，以及依分組屬性（例如 CVflow 的 `teacher`）與最後成敗的狀態比例。
  - **序列重播**：選一條序列逐步檢視，左邊是這次修改後的流程圖與修改理由，右邊是只看前面修改時的路線判斷、各方法的預測與實際修改的排名。

- **生成協助測試**：AoV 生成協助的離線測試。上方是保留測試序列上的整體成績；下方可以選一條測試序列、拖動「已知前幾次修改」，查看系統提出的候選修改、分數的組成、套用後的流程圖，以及實際修改排在第幾名。網址 `#case/<案例代號>/assist` 可直接開啟。
- **轉換設定（提示詞）**：設定本地 LLM 的網址、模型與這個案例的提示詞；可以測試連線。每次修改都會建立新版本。
- **原始文件**：上傳文字檔，並檢視原文內容。
- **轉換與審核**：選擇文件與設定後開始轉換。點選任一筆轉換即可進入審核畫面：
  - 左邊是原文。目前步驟的修改依據以橘色標示，原因引文以綠色標示。
  - 右邊可以逐步查看 AoV 的變化：綠色是新增的活動或依賴，橘色是更新的活動，刪除的部分列在圖下方。另有檢查結果、草稿 JSON（可直接修正）與 LLM 原始回應。
  - 最下方填寫審核人後，可以核准並匯入，或退回。有錯誤時無法核准。

審核畫面的網址為 `#review/<轉換代號>`，可以直接分享給其他審核者。

後端不在 8900 埠時，啟動前端前設定 `KCORE_API_URL`，例如 `$env:KCORE_API_URL="http://localhost:8901"; npm run dev`。

**更新程式後請重新啟動後端**，舊的後端沒有新的 API，畫面會顯示「找不到這項功能的 API」。

## 執行測試

```powershell
cd k_core/backend
python -m pytest
```

## 示範：從示意資料到推論

```powershell
$api = "http://localhost:8900/api/v1"
# 1. 登記場域
Invoke-RestMethod "$api/cases" -Method Post -ContentType "application/json" -Body '{"case_id":"AP-K-ERP","name":"高健","domain_kind":"erp"}'
# 2. 產生 200 條示意演變序列（自動匯入並定案）
Invoke-RestMethod "$api/cases/AP-K-ERP/synthetic-episodes" -Method Post -ContentType "application/json" -Body '{"count":200,"seed":1}'
# 3. 訓練：K 從 2 到 5，以 BIC 選擇（約 30～45 秒）
$run = Invoke-RestMethod "$api/cases/AP-K-ERP/training-runs" -Method Post -ContentType "application/json" -Body '{"k_min":2,"k_max":5}'
$run.metrics.training.selected_k; $run.metrics.test | ConvertTo-Json
# 4. 用使用中的模型，對一條測試序列的前 2 個事件推論下一個修改
$ep = $run.dataset.test_episodes[0]
Invoke-RestMethod "$api/cases/AP-K-ERP/models/active/infer" -Method Post -ContentType "application/json" -Body "{`"episode_id`":`"$ep`",`"prefix_length`":2}"
```

## 用本地 LLM 整理原始資料（以 LM Studio 為例）

1. 在 LM Studio 載入模型，開啟 **Local Server**（預設位址 `http://localhost:1234/v1`）。Ollama、vLLM 等提供 OpenAI 相容介面的服務也可以使用。
2. 為案例建立轉換設定。`system_prompt` 寫這個案例的分析方式，例如資料長什麼樣子、活動類型怎麼對應。K 會自動在後面附上固定的 JSON 格式說明，內容見 `kcore/conversion/draft.py` 的 `FORMAT_INSTRUCTIONS`：

```powershell
$api = "http://localhost:8900/api/v1/cases/AP-K-ERP"
$profile = @{
  profile_id = "apk-default"; name = "AP-K 流程紀錄"
  base_url = "http://localhost:1234/v1"; model = "（LM Studio 顯示的模型名稱）"
  system_prompt = "你是 AP-K 製造流程分析助理。活動類型只能使用 prepare、cut、process、assemble、inspect、pack。"
} | ConvertTo-Json
Invoke-RestMethod "$api/llm-profiles" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($profile))
Invoke-RestMethod "$api/llm-profiles/apk-default/test" -Method Post      # 檢查連線與模型是否存在
```

3. 上傳原始文件。目前只支援文字檔，會自動判斷 UTF-8 或 Big5；同一份內容重複上傳不會建立新紀錄：

```powershell
curl.exe -F "file=@紀錄.txt" "$api/documents"
```

4. 執行轉換。轉換在背景進行，請反覆查詢直到狀態變成 `needs_review` 或 `failed`：

```powershell
$c = Invoke-RestMethod "$api/conversions" -Method Post -ContentType "application/json" -Body '{"document_id":"<document_id>","profile_id":"apk-default"}'
Invoke-RestMethod "$api/conversions/$($c.conversion_id)"   # 看 status、validation、draft、raw_response
```

5. 人工確認：
   - 看 `validation`：`errors` 會阻擋核准，`warnings` 是提醒。
   - 需要修正時：`PUT $api/conversions/{id}/draft`，body 為 `{"draft": {...}}`，修正後會重新驗證。
   - 核准：`POST $api/conversions/{id}/approve`，body 為 `{"reviewer":"姓名"}`。核准後資料匯入並定案，可以用來訓練。
   - 退回：`POST $api/conversions/{id}/reject`。

補充說明：
- **修改提示詞**：`PUT $api/llm-profiles/{profile_id}` 會建立新版本；舊的轉換紀錄仍指向當時的版本。
- **API 金鑰**：如果服務需要金鑰，請用 `api_key_env` 指定環境變數名稱，金鑰本身不會存入資料庫。
- **不支援結構化輸出的服務**：把 `structured_output` 設為 `false`，K 會改用一般回應，再從中擷取 JSON（會自動去除 `<think>` 區塊與程式碼框）。

## 匯入真實資料的流程（已經是 JSON 時）

1. `POST /api/v1/cases/{case_id}/episodes`：提交 `episode_id`、`work_order_id`、`initial_workflow`（`workflow_id`、`version_id`、`nodes`、`edges`）。
2. `POST /api/v1/cases/{case_id}/episodes/{episode_id}/events`：批次提交修改事件，可以分多次送出、不必依照順序。回傳結果會列出已接受、重複與被拒絕的事件。
3. `POST /api/v1/cases/{case_id}/episodes/{episode_id}/finalize`：檢查順序是否連續、版本鏈是否一致、每一步是否都是合法 DAG。通過後保存所有版本快照與衍生特徵。

事件格式與規格書第 9 節的範例相同（`tests/conftest.py` 裡有完整範例）。補充規則：

- `reason` 為選填。有填寫時，`reason_source` 必須是 `human` 或 `llm`。
- 刪除節點時，必須在 `removed_edges` 列出該節點的所有連線。
- `actor_type`：`human`、`system` 或 `synthetic`。

## API 一覽（`/api/v1`）

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| POST／GET | `/cases`、`/cases/{case_id}` | 登記與查詢場域 |
| POST／GET | `/cases/{case_id}/episodes`、`.../{episode_id}` | 建立與查詢演變序列（含快照與事件） |
| POST | `.../{episode_id}/events` | 批次匯入事件 |
| POST | `.../{episode_id}/finalize` | 定案並重建 AoV |
| POST | `/cases/{case_id}/synthetic-episodes` | 產生示意資料 |
| POST／GET | `/cases/{case_id}/training-runs`、`.../{run_id}` | 訓練與查詢結果 |
| GET | `/cases/{case_id}/models`、`.../{model_version_id}` | 模型清單；單一模型含參數、每個狀態的主要觀測、平穩分布 |
| POST | `.../models/{model_version_id}/activate` | 切換使用中的版本 |
| POST | `/cases/{case_id}/aov/assist` | 對某條序列的第 N 步提出候選修改（候選 → HMM 評分 → DAG 驗證），並與實際的下一次修改比對 |
| GET | `/cases/{case_id}/aov/assist-evaluation` | 在模型保留的測試序列上評估候選排名 |
| POST | `/cases/{case_id}/aov/validate` | 檢查一張流程是否為合法的有向無環圖 |
| POST | `.../models/{model_version_id\|active}/replay` | 序列重播：每一步的路線判斷、各方法的預測與實際修改的排名 |
| GET | `.../models/{model_version_id\|active}/analysis?group_by=` | 解碼全部序列，描述每個隱藏狀態與各組比例 |
| POST | `.../models/{model_version_id\|active}/infer` | 前綴推論：每步信念、維特比路徑、下一觀測 Top-k；`method` 可為 `auto`（預設，HMM 與上一次修改各半平均）、`hmm`、`hmm_bigram_mix`、`hmm_route_bigram` |
| POST／GET／PUT | `/cases/{case_id}/llm-profiles`、`.../{profile_id}` | LLM 轉換設定；PUT 建立新版本；GET 可加 `?version=` |
| POST | `.../llm-profiles/{profile_id}/test` | 檢查 LLM 連線與模型 |
| POST／GET | `/cases/{case_id}/documents`、`.../{document_id}` | 上傳（multipart）與查詢原始文件 |
| POST／GET | `/cases/{case_id}/conversions`、`.../{conversion_id}` | 建立背景轉換；查詢草稿、驗證結果、原始回應 |
| PUT | `.../conversions/{conversion_id}/draft` | 審核者修正草稿 |
| POST | `.../conversions/{conversion_id}/approve`、`/reject` | 核准匯入或退回 |

錯誤回應格式：`{"error": {"code", "message", "details"}}`，常見代碼包括 `case_not_found`、`case_mismatch`、`invalid_workflow`、`finalize_failed`、`no_active_model`。訓練資料不足時，訓練工作的狀態會是 `insufficient_data`，不會產生模型。

## 尚未包含

知識項目、AoV 生成協助（`/aov/assist`、`/aov/validate`）、知識項目檢視、背景訓練、非文字檔與長文件的轉換、與 A、C、H、AP 的實際串接。詳見設計文件第 5.2、5.3 節。
