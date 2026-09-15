# P24：容量配置與全段接收服務重播

READY WITH GATES。P23 高吞吐片段顯示20 MHz／2 streams／long GI 參考容量不足，本輪檢查在訓練資料選擇容量設定後，能否改善全段實測服務對照。P22/P23 的資料摘要與結果已揭露；這是明示的後續探索，不稱新獨立確認集。P23 原設定、全部結果與原正文／回覆／47表封存保留。

## 真值與可辨識性

真值仍是同次 WiChoose lustosa5 接收端 application UDP payload bytes/秒。可直接觀測 socket accepted workload 和 sender RSSI；它們可條件化服務比較，但不能單獨識別實體失敗原因或實際硬體配置。兩個不同最大容量的設備，在相同低工作量下可能產生同樣接收量；吻合只能支持有效服務模型，不能反推唯一 hardware。TX 的回壓資訊仍須由直接 TX／簡單校準基線控制。原 fixed-D 所需完整 busy/MAC/C2/BLE/video 不由本轮生成。

## 來源線索與候選

作者較早 Porto 論文描述 Talon AD7200、20 MHz n 平台；WiChoose src README 提及 LEDE-AD7200 編譯目標。原廠 Canada 頁明示2.4/5 GHz 的4-Stream能力：https://www.tp-link.com/ca/home-networking/wifi-router/ad7200/ 。這只是配置候選的依據，不等同 lustosa5 2024-07-19 真實運行設定。ns-3 channel/width設定根據官方文檔及本機3.47 channel table：https://www.nsnam.org/docs/models/html/wifi-user.html 。不以 advertised 800 Mbps 直接作 application 容量。

固定8個標準n候選：20 MHz × streams{2,3,4} × GI{long,short}（6個），40 MHz × streams2 × GI{long,short}（2個）。頻段、中心channel6、Nist、Minstrel-HT、noise7dB、queue512p/0.5秒、A-MPDU65535、retry7、UDP1440bytes uplink、RSSI有效互易及逐秒等距送包均沿用P23。除上述容量參數外，不改任何機率、路徑loss或測試工作量。40MHz維持中心channel6，primary20 index0，作頻寬敏感度。

## 訓練與選擇

保持P23前1556秒train、60秒gap、後978秒evaluation。候選訓練僅用P23預定三段（各20秒，共60秒），每段相同runs23001/23002/23003。共72次訓練模擬。先通過工程與舊配置等價檢查，再鎖定所有候選/輸入/程式，執行72次。

以60個訓練秒之seed平均預測MAE排序；誤差在最小值1%以內視為近似候選，按較窄頻寬、较少streams、long GI優先、名稱字典序選一個。報告全部排序與近似集合，不把選定配置說成唯一實機真值。預先將同一60秒上的非負乘數最小二乘校準保存作次級比較；主比較使用原始封包結果，不用乘數掩蓋容量設定。

選擇完成後先輸出selection_binding，再執行evaluation：只跑選定配置與原20MHz2streamslongGI參考，兩者各3個相同runs（共6次長模擬）。每次連續重播後978秒，前4秒真實輸入暖機、前2秒完成association。保存每秒與逐包接收；不將P23分段重置狀態的數字當成新全段對照組。

## 工程和判準

先重現P23六種工程條件，逐秒/逐包輸出與P23同設定逐位一致。8配置各執行強訊號400Mbps飽和4秒工程測試，確認capacity參數生效、零重複/錯包並記錄各自輸出容量。工程以工作量計數/有效運行為門檻；不要求所有候選一定更好。所有候選與失敗紀錄保留。

主評估為全978秒的分布W1/實測平均（NW1）、seed平均預測MAE/同一平均（NMAE）。逐mobility period、P23原六段和其餘秒均報告；還有原始/訓練乘數校準兩個版本、P23預定選出的rssi_tx_ridge_0.1、TXidentity/TXaffine基線。

強化先導目標：全段NW1≤0.10且NMAE≤0.10，每個原作者period（含未描述尾段）NMAE≤0.20，且相較同段舊配置MAE至少下降50%。這是本輪工程目標，不是審稿標準。相較凍結最強簡單基線再降10%才稱有額外精度優勢。

以60秒circular moving blocks、1000次、seed240914提供配對描述性區間，保留draws；單次實驗且已揭露資料，不作獨立任務或確認性p值宣稱。30秒和120秒block長度一併作預定敏感度，不能只取最有利區間。capacity候選選擇的不確定性不含在evaluation-only區間中，另列near-optimal集合。

## 交付與停止規則

交付硬體來源對照、72次候選訓練/選擇、6次全段replay、全部對照與事件核驗、可支持說法及可重現分析本。若數個配置同樣吻合，只報條件服務有效性，不指認實機；若未優於簡單模型，保留該結果並明確其不同用途。達成全段服務比較後不再追著測試資料調參。完整原後驗缺口不由本輪自動解除。
