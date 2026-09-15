# P19：獨立協定環境的政策與服務參照

本輪是在閱讀 P18 結果後規劃的新實驗。P18 保留封存；P19 的科學設定和分析程式先固定，再使用新 RNG 樣本。目的為評估完整的「可觀測前綴→診斷→候選→驗證→服務排序→一次介入→封包結果」介面。

## 固定問題與範圍

本輪接續既有有限服務模型的 joint-belief 介面，建立新的 ns-3 觀測／服務適配案例。原任務文字、保存的模型候選、語義基線、routing、nominal guards 與 fallback 語義保持固定。新診斷器估計十二種已定義運行條件的概率，服務表只從訓練動作重播估計。這不是原 25 維診斷器或原數值成本的原樣外部重驗，也不是實測部署 posterior。

十二個條件為外部流量 W=0/1、移動速度 S=0/1.0/3.5 m/s、影片 V=平穩/突發的全組合。初始距離保持 15 m，保留 P18 的 3.5 m/s 嚴重條件，加入固定 1 m/s 條件。其餘信道、功率、MAC/PHY、原始提供流量、五個 handler、生效時間、期限及排空期全部沿用 P18。只有一個公開 Parrot 控制流案例；不能將固定條件均值解讀為真實 UAV 場景人口平均。

## 資訊契約

控制端可用的前綴於 10.999999999 s 保存，命令於 11 s 提交、11.001 s 生效。只使用控制端自身 C2 的提供、發送、PHY 嘗試、MAC ACK 和 drop，以及控制端收到的影片時間／位元組和實際成功接收 RSSI。first_ack 是 MAC 確認完成時刻，不是應用層單向時延。

影片接收記錄不包含原始提供需求；提供需求只在評估分母中使用。RSSI 按每次成功收到的影片 MAC frame 等權，保留重傳的成功接收事件；這是接收條件抽樣。packet ID 僅用於測量核驗，不用作特徵。沒有接收樣本的統計量以訓練均值補值並保留缺失指示；若某欄在全部 TRAIN 都缺失，固定填零並保留指示。固定狀態、外部干擾的真實提供量、設定速度／距離、遠端 C2 接收時刻、測試結果均不進入診斷。不能對 final packets 篩選舊 offer 再讀已於介入後更新的欄位。

C2 提供 cohort 與影片接收總量／間隔的支持窗為 [1,10.99) s；影片 100-ms 分箱使用 [1,10.9) s 的 99 個完整接收窗。RSSI 使用保存前綴的全部成功事件，前後段以 6 s 分界。這些支持窗全部位於命令前，各欄並未被解讀為同一種量測。

## 資料分離及執行順序

- 新 seed=6091419；TRAIN runs 1001–1016，共 192 個獨立前綴與 960 次五動作重播。
- VALIDATION runs 2001–2008，共 96 個前綴，僅用於在 C=0.1/1/10 中選 joint log loss 最小者；相同值選較小 C。
- TEST runs 3001–3032：先產生 384 個 prefix-only probes，再保存所有政策的決策和 scores；之後才執行 1,920 次五動作 bank。
- 每一組 action branch 與 probe 的 packet/RSSI 前綴須逐 byte 相同。所有方法選擇的 branch 都有實際 ns-3 執行結果；同一物理 branch 可被多個任務／方法共用，不能把這些重用稱為新的無線實驗。

多類 logistic regression 以 TRAIN 的一個 episode 一份觀测為單位；同前綴的五個動作不能被複製為五份訓練樣本。直接擬合十二狀態的 joint probabilities，service expectation 保留全部十二狀態。qW、qM、qV 由其邊際化，用於原 routing／guard。qB=0 來自本次明確不含 BLE 的受控設定；qM 表示移動條件存在，不等於移動必然造成故障。

## 任務、候選和共同評分

固定納入原 48 OOD 任務中 rid=false 且 energy=false 的全部 19 項。名單依原任務元資料決定；排除清單完整保留。此案例量測 C2/video，沒有 RID/BLE 或能耗模型。所有任務等權；模型生成的三個保存版本也全部等權納入，不能擇優。

P12 的 opaque_zero、public_tool_agent、broad_first3、embedding_first3 候選採相同三個 distinct entry 上限；先截取三個，再做五動作能力過濾，不以第四個補位。完整庫對照明示使用全部五個動作。guarded 方法共用原 nominal guards；候選全部拒絕後只能執行通過同一 guard 的 FallbackProtect，否則為非致動 EscalateReview。Direct 選第一個 capability-valid entry；空列表亦 escalation。Escalation 的物理後果沿用 Observe branch，但 decision label、率和被拒絕狀態不能改寫成已通過驗證的 Observe。

每任務的服務損失為 alpha*C2 miss+(1-alpha)*video nondelivery，alpha=原保存 safety/(safety+throughput)。十九項分母均須大於零。這是 unitless 服務比例，與原 MLU 區分；沒有將任意 overhead/mismatch 係數寫成封包量測成本。每個 action/state 的兩服務均值僅由 TRAIN bank 估計，所有候選來源共用同一個 posterior 期望排序。另把原數值 score 同樣套到五種 guarded 來源作完整描述性 transfer sensitivity。

## 評估與參照

主要結果為全部原始提供 cohort 的 C2 10-ms 期限違反、影片交付比例及上述服務損失。原始影片提供量保留在分母，shaper drop 不消失。失敗、晚到和排空截止仍未收到者都保留；成功收到的延遲另列。期望排序模型不能定義測試服務真值，真值始終來自執行 ledger。

主要比較為 QwenService 對 Direct、EmbeddingService、BroadService、FullLibraryService，各三結果，共十二項。以 32 個 RNG block 為推論單位；每 block 內先平均十二固定條件、十九任務和全部生成版本。exact sign-flip 使用有理數轉整數的 16+16 meet-in-the-middle，雙側、包含等值，null 為符號交換性。10000 次共用 block bootstrap 給點態 percentile 95% CI。全部十二項納入原 101 項，重新給出 pooled113 Holm/BH，保留全部方向。場景、任務、numeric score、工具候選及資料品質是完整描述性結果。

可執行 FullLibraryService 只能用共同前綴、新診斷、TRAIN 服務表及同一 guard。事後參照另外計算全部 A5 最佳，以及同 q-admitted A5 最佳；它們會看到測試分支結果，只是已實現有限 bank 的 retrospective bound，不能稱為部署政策、無偏的最佳平均服務估計或物理世界最優。負的 admitted gap 對未驗證 direct 方法可能成立，不能截零。

## 邏輯核查與停止條件

- 方法刪除後，封包的提供、接收、期限與服務比例仍有獨立定義。
- 相同可見前綴可能對應不同運行條件；診斷保持概率，不用設定真值消除不確定性。
- 測試結果不參與 feature、模型、權重、guard、候選或場景挑選。新試驗不保證某來源勝出。
- hash 或時間隔離失敗、測量分母不守恆、使用不可見資訊時停止正式執行，保留失敗記錄並先修正工程問題；不得以修改結果或刪場景替代。
- 本研究只支持指定適配環境中的一次介入決策比較。BLE共存、原完整實測後驗、硬體時限與開放世界泛化不是此次推論。
