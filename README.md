# initial-research-verification
研究の初期検証

## TensorBoardの開き方
```bash

ssh -L 6006:localhost:6006 ishii@192.168.34.32

cd _iceberg/speaker_direction_estimation

source .venv/bin/activate

tensorboard --logdir runs --port 6006

http://localhost:6006 を開く
