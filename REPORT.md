# Haas-ja-v1.0 検証レポート

2026-10-31 に行った、日本語音声認識モデル [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) の検証結果。

検証した内容は次の 2 つ。

- オリジナルのモデルを Whisper large-v3 / large-v3-turbo と比べた。
- 第三者による ONNX 変換版 2 つを、オリジナルと比べた。
  - [oboroge0/Haas-ja-v1.0-onnx](https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx)(fp32)
  - [oboroge0/Haas-ja-v1.0-onnx-quantized](https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx-quantized)(量子化版)

数値の元データは [`results/`](results/) にある。表は `python bench/summarize.py results` で再計算できる。

## まとめ

- **精度**:
  - Haas の CER(strict)は 4.79%、Whisper large-v3 は 6.50%。Haas が約 26% 少ない。
  - 漢字とかなの書き分けを許容する flex、読みで比べる kana でも、3 つのデータセットすべてで Haas が上回った。
- **速度**:
  - RTX 2060 SUPER(fp16)での RTFx は、Haas が 445、Whisper large-v3 が 31 で、約 14 倍の差がある。
  - 1 件ずつ処理しても(バッチ 1)、Haas 121 に対して Whisper 20 だった。
- **メモリ**:
  - 重みは Haas が 1.2 GB(fp16)、Whisper large-v3 が 2.9 GB。
  - バッチ 1 で処理しているときの使用量は、Haas が約 1.3 GB、Whisper が約 3.3 GB。
- **ONNX 版**:
  - CPU(6 スレッド)で RTFx 19〜24 が出る。CER はオリジナルとの差が 0.1〜0.3 ポイントにとどまる。
  - 量子化版は RAM が約 1 GB で済む。ただし x86 では fp32 版の方が速い。
- **注意点(RTX 2060 SUPER / Turing 世代)**:
  - bf16 はハードウェアで扱えないので、fp16 を使う(fp16 は bf16 の約 3.6 倍速い)。
  - NeMo の CUDA Graph デコーダはバッチ 32 以上で落ちるので、無効にする。

## 環境

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER 8 GB(Turing、compute capability 7.5、ドライバ 615.78.08) |
| CPU | Intel Core i5-13400F(P コア 6 + E コア 4、16 スレッド) |
| RAM | 32 GB |
| OS | CachyOS(Linux 7.3) |
| PyTorch 側 | Python 3.12、torch 2.14.0+cu130、triton 3.8.0、nemo-toolkit 2.7.3、flash-linear-attention 0.5.2、transformers 5.20.2 |
| ONNX 側 | sherpa-onnx 1.13.6 |
| 推論コード | [Haas-ASR](https://github.com/third-intelligence/Haas-ASR) @ `acc614f`(`haas_asr.load_model`) |
| 採点 | [jacer](https://github.com/third-intelligence/jacer) 1.2.0(metrics v1.2: `cer_strict` / `cer_flex` / `cer_kana`) |

## 評価データ

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1 のうち、手元で用意できる次の 3 つを使った。
TEDxJP-10K と SpeechBSD は、元の音声を集める手間が大きいため使っていない。

| subset | 件数 | 音声の長さ | 内容 |
|---|---|---|---|
| JSUT basic5000 | 5000 | 6.8 h | 読み上げ(1 人の女性) |
| JVNV | 1375 | 3.3 h | 感情を込めた発話(悲鳴や笑い声などを含む) |
| FLEURS ja_jp test | 650 | 2.4 h | Wikipedia の文の読み上げ(複数の話者) |
| 計 | 7025 | 12.5 h | |

データは `bench/prep.py` で作った。bench リポジトリの `samples.jsonl` に従って修正を当て、書き起こしのハッシュを照合してから、16 kHz の FLAC にしている。
bench リポジトリにある `rebuild.py` は、Claude Code の auto mode が外部スクリプトの実行を許可しなかったため使っていない。代わりに同じ処理を書き直した。

全体の CER は、各データセットの値を音声の長さで重み付けして平均したもの。RTFx は、音声の総時間を処理時間の合計で割ったもの。
処理時間は推論だけを測っている。音声ファイルの読み込みとモデルのロードは含まない。

## 結果 1: Haas と Whisper(GPU)

条件:

- Haas は fp16、バッチ 32、CUDA Graph デコーダ無効。
- Whisper は fp16、バッチ 8、SDPA、greedy(`language="ja"`)。バッチ 16 だとクロスアテンションの KV キャッシュで 8 GB に収まらない。
- どちらも `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定した。

### 全体

| モデル | CER strict | flex | kana | RTFx | 重み | 推論中の VRAM(torch) | 推論中の VRAM(nvidia-smi) |
|---|---|---|---|---|---|---|---|
| **Haas-ja-v1.0** fp16 | **4.79** | **2.71** | **1.79** | **445** | 1.2 GB | 2.9〜4.1 GB | 3.7〜4.8 GB |
| Haas-ja-v1.0 bf16 | 4.80 | 2.71 | 1.79 | 125 | 1.2 GB | 同上 | 同上 |
| Whisper large-v3 fp16 | 6.50 | 4.33 | 2.62 | 31 | 2.9 GB | 5.0〜5.4 GB | 5.2〜5.8 GB |
| Whisper large-v3-turbo fp16 | 10.37 | 8.18 | 5.91 | 48 | 1.5 GB | 1.9 GB | 2.1 GB |

パラメータ数は Haas が 616M、Whisper large-v3 が 1,543M、turbo が 809M。
VRAM は、torch が確保した量のピーク(`max_memory_allocated`)と、nvidia-smi で見えるプロセス全体の量のピークを並べた。後者には CUDA コンテキストとアロケータのキャッシュも含まれる。

### データセット別(CER strict / flex / kana、RTFx)

| モデル | JSUT | JVNV | FLEURS |
|---|---|---|---|
| Haas fp16 | 5.32 / 2.43 / 1.36、411 | 4.51 / 3.06 / 2.25、554 | 3.68 / 3.00 / 2.38、431 |
| Whisper large-v3 | 6.86 / 3.73 / 1.77、25 | 6.24 / 4.98 / 3.49、41 | 5.82 / 5.14 / 3.81、49 |
| Whisper large-v3-turbo | 7.00 / 3.85 / 1.85、38 | 20.50 / 19.16 / 15.05、60 | 5.78 / 5.14 / 4.69、92 |

### 1 件ずつ処理した場合(バッチ 1、FLEURS)

マイク入力のように 1 件ずつ処理する場面を想定した。

| モデル | CER strict | RTFx | 推論中の VRAM(torch) |
|---|---|---|---|
| Haas fp16 | 3.66 | 121 | 1.3 GB |
| Whisper large-v3 fp16 | 5.82 | 20 | 3.3 GB |

### 認識結果の傾向

`results/hyp/` の発話ごとの結果を見て分かったこと。

- **turbo の JVNV の CER が高い原因**: 15 件で「きゃああああ…」のように同じ文字を出し続ける暴走が起きていた。large-v3 でも、同じ現象が JVNV で 1 件(「うううう…」)起きた。Haas では、出力が正解の 2 倍以上の長さになった発話は 0 件だった。
- **Haas の誤り**: 「一言居士」→「一元孤児」、「才子多病、佳人薄命」→「祭祀多病歌人革命」のような、珍しい四字熟語の同音異義語が多い。
- **Whisper の誤り**: 「豪勢」→「ゴーセイ」、「災禍」→「サイカ」のように、漢字にせずカタカナで書くことがある。

### 精度(dtype)と設定の比較(FLEURS、`results/fleurs_sweep.jsonl`)

| 設定 | RTFx | nvidia-smi のピーク |
|---|---|---|
| bf16、バッチ 8、CUDA Graph あり | 148 | 6.8 GB |
| bf16、バッチ 8、CUDA Graph なし | 130 | 6.7 GB |
| bf16、バッチ 32、CUDA Graph なし | 127 | 7.0 GB |
| fp32、バッチ 8、CUDA Graph なし | 173 | 6.5 GB |
| fp16、バッチ 32、CUDA Graph なし | 423 | 7.0 GB |
| fp16、バッチ 32、CUDA Graph なし、`expandable_segments` | 431 | 4.8 GB |

- Turing 世代は bf16 をハードウェアで扱えず、bf16 は fp32 よりも遅かった。fp16 にすると約 3.6 倍速くなり、CER は変わらない(4.79 と 4.80)。
- CUDA Graph デコーダを有効にすると、バッチ 8 では 14% 速い。ただしバッチ 32 以上では `CUDA error: an illegal memory access was encountered` で落ちる(bf16 でも fp16 でも起きる。`bench/cuda_graph_repro.py` で再現できる)。
- `expandable_segments:True` にすると、メモリの断片化が減って、ピークが 7.0 GB から 4.8 GB に下がった。
- `load_model` の直後は、fp32 の重みが循環参照の中に残っていて、約 1.2 GB 余分に使っている。`gc.collect()` と `torch.cuda.empty_cache()` で、2.4 GB が 1.2 GB に下がる。
- バッチ 64 では 8 GB に収まらない。

## 結果 2: ONNX 変換版(CPU)

条件:

- sherpa-onnx の `OfflineRecognizer.from_transducer` を、`model_type="nemo_transducer"`、`feature_dim=128` で使った。
- 6 スレッドで、1 件ずつ処理した。
- RAM のピーク(RSS)にはモデルと推論時のメモリを含む。データセットの音声と jacer の辞書は含まないように測った(音声は 1 件ずつ読み込み、採点は全件の認識が終わってから行った)。

### 全体

| モデル | CER strict | flex | kana | RTFx | ロード後の RAM | RAM のピーク | ファイルサイズ |
|---|---|---|---|---|---|---|---|
| オリジナル(GPU、fp16) | 4.79 | 2.71 | 1.79 | 445 | — | — | 1.2 GB(.nemo) |
| ONNX fp32 | 5.06 | 2.78 | 1.78 | 23.8 | 2.2 GB | 2.8〜3.1 GB | 2.4 GB |
| ONNX 量子化版 | 4.89 | 2.78 | 1.82 | 18.6 | 0.6 GB | 1.0〜1.3 GB | 459 MB |

### データセット別(CER strict / flex / kana)

| モデル | JSUT | JVNV | FLEURS |
|---|---|---|---|
| オリジナル | 5.32 / 2.43 / 1.36 | 4.51 / 3.06 / 2.25 | 3.68 / 3.00 / 2.38 |
| ONNX fp32 | 5.32 / 2.44 / 1.36 | **5.42** / 3.24 / 2.24 | 3.80 / 3.09 / 2.36 |
| ONNX 量子化版 | 5.31 / 2.48 / 1.38 | **4.82** / 3.15 / 2.29 | 3.81 / 3.13 / 2.43 |

- **JSUT と FLEURS**: オリジナルとほぼ同じ精度。
- **JVNV の strict が悪化する理由**: 感情を込めた発話で、漢字をひらがなで書くことが増える。たとえば「伝染病が猛威を振るっている」が「猛威をふるっている」に、「友達とも会えなくて」が「ともだちともあえなくて」になる。
  - 読みが正しいかを見る kana の CER は、オリジナルとほぼ同じなので、聞き取り自体はできている。
  - 原因は、sherpa-onnx が音声の特徴量を NeMo とは別の実装で計算していることだと推測している(確かめてはいない)。
  - 量子化版の JVNV の出力は、1375 件中 756 件がオリジナルと完全に一致した(`bench/compare_jvnv.py`)。
- **x86 CPU では fp32 版の方が量子化版より速い**: 23.8 と 18.6。量子化の利点は、メモリとファイルサイズが小さいことにある。

### スレッド数と速度(量子化版、FLEURS 32 件)

| スレッド数 | 1 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| RTF | 0.194 | **0.051** | 0.076 | 0.111 | 0.133 |

P コアの数と同じ 6 スレッドが最も速い。それより増やすと処理が遅い E コアにも回り、かえって遅くなる。

## 結果 3: Playground(Web アプリ)での体感速度

[`app/`](app/) の Playground で測った。

**録音・ファイルの認識**: JSUT の 20 文をつないだ 70 秒の M4A を送り、処理時間を測った。

| モデル | 処理時間 |
|---|---|
| オリジナル(GPU) | 0.43 秒 |
| ONNX 量子化版 | 3.6 秒 |
| ONNX fp32 | 2.7 秒 |

**リアルタイム字幕**: FLEURS の 1 文を実時間で WebSocket に流した(`bench/ws_latency.py`)。
遅延は、フレームを送ってから、そのフレームまでを含む認識結果が返ってくるまでの時間。

| モデル | 遅延(中央値 / 最大) | うちモデルの処理(中央値 / 最大) |
|---|---|---|
| オリジナル(GPU) | 125 / 174 ms | 68 / 114 ms |
| ONNX 量子化版 | 325 / 625 ms | 281 / 535 ms |
| ONNX fp32 | 305 / 561 ms | 251 / 493 ms |

Playground は 0.25 秒ごとに、話している途中の文を頭から認識し直す。ONNX 版は文が長くなるほど処理時間が延び、10 秒を超える文では遅れが目立つ。

**サーバー全体のメモリ**: 3 つのモデルをすべて読み込むと、RAM 約 6.7 GB、VRAM 約 2.9 GB を使う。

**実際に使った感想**: マイクで試したところ、「認識は速いし、精度も良い方」だった(GPU 版)。
