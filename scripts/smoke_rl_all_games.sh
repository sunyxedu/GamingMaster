#!/usr/bin/env bash
# RL smoke of every training game through the joint RL trainer: a tiny 21-game Stage-1 checkpoint
# (DATA/<game>/ = one real shard each, e.g. one validation shard per game), then for each game
# train_rl_joint 2 iterations on 2 envs (whole episodes -> GAE -> PPO -> rl_final.pt).
#
#   scripts/smoke_rl_all_games.sh DATA OUT [GAMES...]
#   env: ENVIRONMENTS_DIR (arc3's cached game files); sc2 / hok need their game binaries (cluster).
# On CPUs without oneDNN / NNPACK support (e.g. SSE2-only VMs) sc2's map convolutions are very slow:
# run it with --env_kw sc2:max_loops=400 and torch.backends.mkldnn.enabled = False.
set -uo pipefail
DATA=${1:?DATA}; OUT=${2:?OUT}; shift 2
ALL="generals chess go shogi othello mahjong lux clash_royale gomoku xiangqi chess_variants bridge skat doudizhu hunl backgammon pokemon sc2 arc3 hok tetris"
GAMES=${*:-$ALL}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CORE=$(uv run --no-sync python -c "from tests.test_core_moe import CORE; print(' '.join(f'{k}={v}' for k, v in CORE.items()))")
if [ ! -f "$OUT/ck/multi_final.pt" ]; then
  A=(); for g in $ALL; do A+=(--data "$g=$DATA/$g" --game_max_len "$g=32" --game_val_max_len "$g=32"); done
  uv run --no-sync python -m training.train_multi_game "${A[@]}" --max_len 32 --steps 2 --batch_size 1 \
    --warmup_steps 1 --moe_balance_rate 0.01 --save_dir "$OUT/ck" --device cpu --override $CORE || exit 1
fi
for g in $GAMES; do
  extra=(); [ "$g" = generals ] && extra=(--env_kw generals:max_turns=100)
  [ "$g" = sc2 ] && extra=(--env_kw sc2:max_loops=400)
  log="$OUT/rl-$g.log"
  uv run --no-sync python -m training.train_rl_joint --init "$OUT/ck/multi_final.pt" --rl "$g" --iterations 2 \
    --tokens_per_iter 64 --envs "$g=2" --max_wave_steps 20000 --epochs 1 --minibatches 1 --device cpu --seed 1 \
    --save_dir "$OUT/rl-$g" "${extra[@]}" > "$log" 2>&1
  if grep -q '^\[done\]' "$log"; then
    echo "$g OK $(grep -E "^\s+$g: quota" "$log" | tail -1 | grep -oE 'episodes=[0-9]+|ep_len=[0-9.]+' | tr '\n' ' ')"
  else
    echo "$g FAIL (see $log): $(grep -E 'Error' "$log" | tail -1 | cut -c1-200)"
  fi
done
