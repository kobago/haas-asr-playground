# results

`bench/` のスクリプトが書き出した評価結果。1 行が 1 回の評価(1 モデル × 1 データセット)。

| ファイル | 内容 |
|---|---|
| `results.jsonl` | GPU: Haas fp16 / bf16(バッチ 32)、Whisper large-v3 / large-v3-turbo(バッチ 8) |
| `bs1.jsonl` | GPU: バッチ 1(FLEURS) |
| `fleurs_sweep.jsonl` | GPU: Haas の dtype・バッチ・CUDA Graph デコーダの有無の比較(FLEURS)。最後の行以外は `expandable_segments` なし |
| `onnx_results.jsonl` | CPU: ONNX fp32 / 量子化版(sherpa-onnx、6 スレッド) |
| `hyp/<model>_<dtype or threads>_<subset>.jsonl` | 発話ごとの `{"id", "ref", "hyp"}`(正解と認識結果) |

主なフィールド:

| フィールド | 意味 |
|---|---|
| `audio_s` / `elapsed_s` | 音声の総時間 / 推論にかかった時間(秒) |
| `rtfx` | `audio_s / elapsed_s`。音声の何倍の速さで処理できたか |
| `cer_strict` / `cer_flex` / `cer_kana` | jacer 1.2.0 の CER(%)。strict は送り仮名などの細かい表記の違いだけを許容する。flex は同じ語であれば漢字・ひらがな・カタカナの書き分けも許容する(お願い/おねがい)。kana は OpenJTalk で読みに変換して比べるので、同音異義語(機関/期間)も正解として扱う |
| `weights_MiB` | ロード後(`gc.collect()` の後)に GPU 上にある重みの量 |
| `load_peak_MiB` | ロード中の GPU メモリのピーク |
| `peak_alloc_MiB` / `peak_reserved_MiB` | 推論中の torch の確保量 / 予約量のピーク |
| `peak_nvsmi_MiB` | nvidia-smi で見えるプロセス全体の GPU メモリのピーク |
| `peak_rss_MiB` | プロセスの RAM のピーク。GPU の評価では読み込んだ音声も含む |
| `rss_after_load_MiB` | ONNX 版: モデルをロードした直後の RAM |

`hyp/` の `ref` は、元のコーパスの書き起こしに asr-ja-bench の修正を当てたもの。それぞれのライセンスは次のとおり。

- JSUT: CC BY-SA 4.0(高道慎之介ほか)
- JVNV: CC BY-SA 4.0(Detai Xin ほか)
- FLEURS: CC BY 4.0(Google)
