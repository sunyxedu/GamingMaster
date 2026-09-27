#!/usr/bin/bash
# Launch one HoK simulator under the portable Wine (called by gamecore-server-linux-amd64
# as --simulator-remote-bin / --simulator-repeat-bin; $0's name picks the exe).
conf=$1
new_conf=${conf}.new
work_dir=$(pwd)
set -e
HOK=${HOK:-$HOME/hok}
$HOK/venv38/bin/python $HOK/hok_env/aiarena/remote-gc-server/process.py $conf $new_conf ${work_dir}
GAMECORE_PATH=${GAMECORE_PATH:?}
export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"
export WINEPREFIX=${WINE_PREFIX_ROOT:-/tmp/$USER-wine}/${KAIWU_RUNTIME_ID:-default}
export WINEDEBUG=-all
mkdir -p $WINEPREFIX
exe=sgame_simulator_remote_zmq.exe
case "$(basename $0)" in *repeat*) exe=sgame_simulator_repeated_zmq.exe;; esac
exec $HOK/wine-root/opt/wine-stable/bin/wine64 ${GAMECORE_PATH}/bin/$exe $new_conf
