# haas-asr-playground

日本語音声認識モデル [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) を検証した記録と、マイクで試せる Web アプリ。

- **[REPORT.md](REPORT.md)**: 検証結果。精度・速度・メモリを Whisper large-v3 や ONNX 変換版と比べた。
- **[app/](app/)**: Playground(SPA + API サーバー)。リアルタイム字幕・録音・ファイルの 3 つの画面があり、オリジナル(GPU)と ONNX 版 2 つを切り替えられる。
- **[bench/](bench/)**: 評価用のスクリプト。
- **[results/](results/)**: 評価結果の生データ。

## 構成

```
REPORT.md                 検証レポート
app/
  server.py               API サーバー(FastAPI)。モデルごとに 1 スレッドで推論
  start.sh                起動スクリプト(0.0.0.0:7860、Tailscale の証明書で HTTPS)
  static/                 SPA(index.html / app.js / style.css / capture-worklet.js)
bench/
  prep.py                 asr-ja-bench の JSUT / JVNV / FLEURS を 16 kHz FLAC にする
  bench.py                Haas / Whisper を GPU で評価(CER・RTFx・VRAM・RSS)
  bench_onnx.py           ONNX 版を sherpa-onnx で評価(CPU)
  summarize.py            results/ から REPORT.md の表を作る
  compare_jvnv.py         JVNV でオリジナルと ONNX 版の出力を発話ごとに比べる
  cuda_graph_repro.py     NeMo の CUDA Graph デコーダがバッチ 32 以上で落ちる問題の再現
  ws_latency.py           Playground のリアルタイム字幕の遅延を測る
results/
  results.jsonl           GPU: Haas fp16/bf16、Whisper large-v3 / turbo(バッチ 32 / 8)
  bs1.jsonl               GPU: バッチ 1(FLEURS)
  fleurs_sweep.jsonl      GPU: Haas の dtype・バッチ・CUDA Graph の比較(FLEURS)
  onnx_results.jsonl      CPU: ONNX fp32 / 量子化版(6 スレッド)
  hyp/                    発話ごとの認識結果(id / 正解 / 認識結果)
```

## 前提

- [Haas-ASR](https://github.com/third-intelligence/Haas-ASR) を `~/projects/github.com/Haas-ASR` に clone し、その環境で動かす。
  - kaldialign が Python 3.14 に対応していないので、環境は Python 3.12 で作る。
  - app と bench は、どちらも Haas-ASR の環境で動く(Haas-ASR のファイルは変更しない)。
  ```bash
  cd ~/projects/github.com/Haas-ASR && uv sync --python 3.12 --group eval
  ```
- `/tmp` は 16 GB の tmpfs で、NeMo が `.nemo` を展開すると容量が足りなくなる。そのため `TMPDIR=~/.cache/haas-asr-eval/tmp` を使う。
- ffmpeg が必要。

## Playground を起動する

```bash
app/start.sh   # https://cachyos-x8664.tail3edcfb.ts.net:7860
```

詳しくは [app/README.md](app/README.md) を参照。

## 評価を再現する

以下は、このリポジトリの直下で実行する。

```bash
W=~/.cache/haas-asr-eval
H=~/projects/github.com/Haas-ASR
mkdir -p $W/corpora $W/tmp

# 1. bench の定義(samples.jsonl)と元のコーパスを取ってくる
$H/.venv/bin/hf download third-intelligence/asr-ja-bench --repo-type dataset --local-dir $W/bench-repo
(cd $W/corpora && curl -sSfLO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip \
                && curl -sSfLO https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip)

# 2. 評価データを作る(FLEURS は Hugging Face から自動で取ってくる)
$H/.venv/bin/python -I bench/prep.py --data $W/bench-repo/data --out $W/bench \
    --subset jsut --subset jvnv --subset fleurs \
    --jsut $W/corpora/jsut_ver1.1.zip --jvnv $W/corpora/jvnv_ver1.zip

# 3. GPU の評価(モデルごとに別のプロセスで動かし、メモリの数値が混ざらないようにする)
export TMPDIR=$W/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$PWD
run() { (cd $H && uv run python $P/bench/bench.py --bench $W/bench \
         --subset jsut --subset jvnv --subset fleurs --out $P/results/results.jsonl \
         --hyp-dir $P/results/hyp "$@"); }
run --model haas --dtype fp16 --batch-size 32 --no-cuda-graph
run --model haas --dtype bf16 --batch-size 32 --no-cuda-graph
run --model whisper --dtype fp16 --batch-size 8
run --model whisper-turbo --dtype fp16 --batch-size 8

# 4. ONNX 版の評価(CPU 6 スレッド)
for r in "oboroge0/Haas-ja-v1.0-onnx fp32" "oboroge0/Haas-ja-v1.0-onnx-quantized q"; do
  set -- ${=r}   # zsh。bash なら set -- $r
  $H/.venv/bin/hf download $1 --exclude "conversion/*" --local-dir $W/onnx/$2
done
R="uv run --no-project --python 3.12 --with sherpa-onnx==1.13.6 --with soundfile \
   --with jacer@git+https://github.com/third-intelligence/jacer.git@b861959fe0a8f67f47c848d19c1a9cb8a32e1c2e python -I"
${=R} bench/bench_onnx.py --model-dir $W/onnx/fp32 --encoder encoder.onnx --label onnx-fp32 \
    --bench $W/bench --subset jsut --subset jvnv --subset fleurs --threads 6 \
    --out results/onnx_results.jsonl --hyp-dir results/hyp
${=R} bench/bench_onnx.py --model-dir $W/onnx/q --encoder encoder.quantized.onnx --label onnx-q \
    --bench $W/bench --subset jsut --subset jvnv --subset fleurs --threads 6 \
    --out results/onnx_results.jsonl --hyp-dir results/hyp

# 5. 表にする
python3 bench/summarize.py results
```

所要時間の目安(RTX 2060 SUPER / i5-13400F):

| 評価 | 時間 |
|---|---|
| Haas fp16 | 2 分 |
| Haas bf16 | 6 分 |
| Whisper large-v3 | 24 分 |
| Whisper large-v3-turbo | 16 分 |
| ONNX fp32 | 32 分 |
| ONNX 量子化版 | 40 分 |

使うディスク容量は次のとおり。

- コーパスの zip: 4.5 GB
- 評価データ: 0.7 GB
- ONNX 版: 2.9 GB
- Hugging Face のキャッシュ(モデル): 別途

## ライセンス

- **コードとドキュメント**: [Apache License 2.0](LICENSE)。Copyright 2026 kobago
- **`app/static/capture-worklet.js`**: [Haas-ASR](https://github.com/third-intelligence/Haas-ASR) の `demos/static/capture-worklet.js` をそのまま使っている(Apache License 2.0)。
- **`results/hyp/`**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。
  - 正解文(`ref`)は、評価用コーパスの書き起こしに [asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) の修正を当てたもの。
  - 元のコーパスのライセンスを引き継いでいる。詳しくは [results/README.md](results/README.md) を参照。
