# Haas Playground

Haas-ja-v1.0 をマイク入力などで試すための SPA + API サーバー。
画面右上でモデルを切り替えられる。

| id | モデル | 実行環境 |
|---|---|---|
| `haas-gpu` | オリジナル([third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)) | GPU、PyTorch + NeMo、fp16 |
| `onnx-q` | [oboroge0/Haas-ja-v1.0-onnx-quantized](https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx-quantized) | CPU 6 スレッド、sherpa-onnx |
| `onnx-fp32` | [oboroge0/Haas-ja-v1.0-onnx](https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx) | CPU 6 スレッド、sherpa-onnx |

ONNX 版のファイルは初回起動時に `~/.cache/haas-mic-app/models/` へダウンロードされる(約 2.9 GB)。
Hugging Face の通常のキャッシュを使わないのは、そこでは `encoder.onnx` と `encoder.data` が別々のフォルダにある実ファイルへのシンボリックリンクになり、
ONNX Runtime が「外部データがモデルのフォルダの外にある」として読み込みを拒否するため。
`--onnx`(引数なし)を付けると、ONNX 版を読み込まずに起動する。

推論部分は [Haas-ASR](https://github.com/third-intelligence/Haas-ASR) リポジトリ(`~/projects/github.com/Haas-ASR` に clone したもの。場所は環境変数 `HAAS_REPO` で変えられる)の `haas_asr` パッケージと、`demos/realtime.py`(リアルタイム字幕の WebSocket と音声の区切り処理)を使う。

- **リアルタイム字幕**: マイク音声を WebSocket で送り、0.25 秒ごとに認識し直して表示する。0.7 秒の無音で確定する。
- **録音して認識**: 録音を止めた後、WAV にまとめて `POST /api/transcribe` で認識する。
- **ファイル**: ffmpeg が読める形式(WAV / MP3 / M4A / WebM / 動画など)をアップロードして認識する。
- **全モデルで比較**(録音・ファイルの画面): 同じ音声を全モデルで順番に認識し、結果と処理時間を並べて表示する。
  同時に送らないのは、CPU で動く ONNX 版 2 つがコアを取り合い、処理時間を公平に比べられなくなるため。

## 起動

```bash
./start.sh   # https://cachyos-x8664.tail3edcfb.ts.net:7860
```

ブラウザがマイクを使わせてくれるのは HTTPS か localhost だけなので、Tailscale の証明書で HTTPS を有効にしている
(`tailscale cert` で取得。証明書は `~/.cache/haas-mic-app/certs`、有効期限は約 90 日なので、切れたら同じコマンドで取り直す)。
IP アドレスではなく、`https://cachyos-x8664.tail3edcfb.ts.net:7860` の形で開くこと。

## API

| | |
|---|---|
| `GET /api/info` | 使えるモデルの一覧(id、表示名、実行環境、精度) |
| `POST /api/transcribe?model=ID` | リクエストボディに音声ファイル → `{model, text, segments: [{start, end, text}], duration, elapsed, rtfx}` |
| `WS /ws/realtime?model=ID` | Haas-ASR の `demos/realtime.py` と同じプロトコル |

`model` を省略すると `haas-gpu` を使う。

```bash
curl -X POST --data-binary @example.m4a "https://cachyos-x8664.tail3edcfb.ts.net:7860/api/transcribe?model=onnx-q"
```

## RTX 2060 SUPER(8 GB / Turing 世代)向けの設定

- **fp16 で推論する**: Turing 世代は bf16 をハードウェアで扱えないため、bf16 だと約 3.6 倍遅い。
- **NeMo の CUDA Graph デコーダを無効にする**: 有効なままバッチ 32 以上で処理すると illegal memory access で落ちる。
- **バッチは最大 16**: 8 GB に収まるようにするため。
- **ロード後に `gc.collect()` する**: ロード直後に残る fp32 の重み(約 1.2 GB)を解放するため。
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: メモリの断片化によるピークの増加を抑えるため。

ONNX 版は 6 スレッドで動かす。i5-13400F の P コアの数で、これより増やすと遅い E コアにも処理が回り、かえって遅くなる(`--onnx-threads` で変更できる)。

認証はないので、Tailscale のネットワーク内だけで使うこと。
