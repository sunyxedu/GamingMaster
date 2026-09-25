#!/bin/bash
# Shared-core study on hephaestus: one code snapshot, one hyper-parameter
# set; every run differs from the base only in the factor under study.
#
#   SNAP=~/gm-snap/<sha> STEPS=8000 bash scripts/slurm/multi/submit_core_study.sh q1   # cores
#   SNAP=~/gm-snap/<sha> STEPS=8000 bash scripts/slurm/multi/submit_core_study.sh q2   # single-game
#   BRAIN=<ckpt> bash scripts/slurm/multi/submit_core_study.sh q3  # held-out adaptation
#
# Q1  joint 5-game IL, core ∈ {transformer-6, transformer-12, recurrent-depth,
#     recurrent-depth+MoE, transformer-6+MoE}
# Q2  each game alone with the recurrent-depth+MoE core, trained on exactly as
#     many micro-batches of that game as the joint run gives it
#     (steps_g = STEPS · K · w_g, K = 5) — positive/negative transfer test
# Q3  adapt the Q1 brain to held-out games (shogi, pokemon):
#     frozen core | frozen core + trainable norms | full fine-tune | scratch
set -euo pipefail
STUDY=${1:?q1|q2|q3}
STEPS=${STEPS:-8000}
GRES=${GRES:-gpu:nvidia_h200_nvl:1}
SNAP=${SNAP:?set SNAP to a gm-snap worktree}
D=${DATA_ROOT:-$HOME/gm-data/split}
GAMES=(othello chess go mahjong clash_royale)

COMMON="--batch_size 8 --game_batch clash_royale=2 --max_len 64 --bf16 --num_workers 4 \
--lr 3e-4 --warmup_steps 500 --log_every 100 --eval_every 1000 --eval_batches 40 --seed 0"
BASE="d_model=256 n_heads=8 d_ff=1024 n_st_layers=2 dropout=0.0 temporal_mixer=recurrent \
core_expansion=4.0 rd_prelude_layers=2 rd_recurrent_layers=2 rd_coda_layers=2 \
rd_mean_iters=8 rd_backprop_iters=4 rd_max_iters=32"
RDMOE="core_type=recurrent_depth moe_experts=8 moe_top_k=2 moe_shared_experts=1 moe_expansion=2.0"

data_args() {
    local out=""
    for g in "$@"; do out="$out --data $g=$D/$g/train --val $g=$D/$g/val"; done
    echo "$out"
}

submit() {   # RUN ARGS [extra sbatch flags...]
    local run=$1 args=$2; shift 2
    GM_HOME=$SNAP RUN=$run ARGS="$args" sbatch --requeue --gres="$GRES" "$@" \
        "$SNAP/scripts/slurm/multi/multi_game_job.sh"
}

case $STUDY in
q1)
    J=$(data_args "${GAMES[@]}")
    submit q1-tf6     "$J $COMMON --steps $STEPS --override $BASE core_type=transformer core_layers=6"
    submit q1-tf12    "$J $COMMON --steps $STEPS --override $BASE core_type=transformer core_layers=12"
    submit q1-rd      "$J $COMMON --steps $STEPS --override $BASE core_type=recurrent_depth"
    submit q1-rdmoe   "$J $COMMON --steps $STEPS --override $BASE $RDMOE"
    submit q1-tf6moe  "$J $COMMON --steps $STEPS --override $BASE core_type=transformer core_layers=6 moe_experts=8 moe_top_k=2 moe_shared_experts=1 moe_expansion=2.0"
    ;;
q2)
    # the joint run's mixture weights, computed exactly as train_multi_game
    # does (alpha = 0.5 on frames capped at --max_len; clash_royale uncropped)
    for g in "${GAMES[@]}"; do
        S=$(cd "$SNAP" && .venv/bin/python -c "
from data_processing.games.dataset import GameShardDataset
from training.train_il_game import UNCROPPABLE
from training.train_multi_game import mixture_weights
games = '${GAMES[*]}'.split()
fr = {x: GameShardDataset('$D/' + x + '/train', game=x,
                          max_len=None if x in UNCROPPABLE else 64).total_frames() for x in games}
print(round($STEPS * len(games) * mixture_weights(fr, 0.5)['$g']))")
        echo "q2-$g: $S steps"
        submit "q2-$g" "$(data_args "$g") $COMMON --steps $S --games_per_step 1 --override $BASE $RDMOE"
    done
    ;;
q3)
    B=${BRAIN:?set BRAIN to the q1 brain checkpoint}
    AS=${ADAPT_STEPS:-4000}
    for g in shogi pokemon; do
        A="$(data_args "$g") $COMMON --steps $AS --eval_every 500"
        submit "q3-$g-frozen"      "$A --init_core $B --freeze_core"
        submit "q3-$g-frozennorm"  "$A --init_core $B --freeze_core --train_core_norms"
        submit "q3-$g-finetune"    "$A --init_core $B"
        submit "q3-$g-scratch"     "$A --override $BASE $RDMOE"
    done
    ;;
*) echo "unknown study $STUDY" >&2; exit 2 ;;
esac
