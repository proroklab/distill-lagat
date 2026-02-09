#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JOBS="${JOBS:-8}"
TARGET="all"
DO_CLEAN=0
DO_DISTCLEAN=0

usage() {
  cat <<'EOF'
Usage: scripts/build_cpp.sh [--lacam3|--lagat|--interfaces|--planner-lacam3|--planner-lagat|--planners|--all] [--clean|--distclean]

Examples:
  scripts/build_cpp.sh --all
  scripts/build_cpp.sh --lacam3
  scripts/build_cpp.sh --lagat --clean
  scripts/build_cpp.sh --planners
  scripts/build_cpp.sh --planner-lacam3
  scripts/build_cpp.sh --planner-lagat --clean
  scripts/build_cpp.sh --distclean
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lacam3) TARGET="interface-lacam3" ;;
    --lagat) TARGET="interface-lagat" ;;
    --interfaces) TARGET="interfaces" ;;
    --planner-lacam3) TARGET="planner-lacam3" ;;
    --planner-lagat) TARGET="planner-lagat" ;;
    --planners) TARGET="planners" ;;
    --all) TARGET="all" ;;
    --clean) DO_CLEAN=1 ;;
    --distclean) DO_DISTCLEAN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

preset_for_target() {
  case "$1" in
    interface-lacam3) echo "interface-lacam3" ;;
    interface-lagat) echo "interface-lagat" ;;
    *) echo "Unknown target: $1" >&2; exit 1 ;;
  esac
}

build_dir_for_target() {
  case "$1" in
    interface-lacam3) echo "${REPO_ROOT}/build/interface-lacam3" ;;
    interface-lagat) echo "${REPO_ROOT}/build/interface-lagat" ;;
    planner-lacam3) echo "${REPO_ROOT}/build/lacam3" ;;
    planner-lagat) echo "${REPO_ROOT}/build/lagat" ;;
    *) echo "Unknown target: $1" >&2; exit 1 ;;
  esac
}

source_dir_for_target() {
  case "$1" in
    interface-lacam3) echo "${REPO_ROOT}/src/lagat/interfaces/lacam3" ;;
    interface-lagat) echo "${REPO_ROOT}/src/lagat/interfaces/lagat" ;;
    planner-lacam3) echo "${REPO_ROOT}/cpp_planners/lacam3" ;;
    planner-lagat) echo "${REPO_ROOT}/cpp_planners/lagat" ;;
    *) echo "Unknown target: $1" >&2; exit 1 ;;
  esac
}

run_clean() {
  local target="$1"
  local build_dir
  build_dir="$(build_dir_for_target "${target}")"
  if [[ -d "${build_dir}" ]]; then
    cmake --build "${build_dir}" --target clean
  else
    echo "Skip clean; build dir missing: ${build_dir}"
  fi
}

run_distclean() {
  local target="$1"
  local build_dir
  build_dir="$(build_dir_for_target "${target}")"
  if [[ -d "${build_dir}" ]]; then
    rm -rf "${build_dir}"
  else
    echo "Skip distclean; build dir missing: ${build_dir}"
  fi
}

run_build() {
  local target="$1"
  local source_dir
  source_dir="$(source_dir_for_target "${target}")"
  local build_dir
  build_dir="$(build_dir_for_target "${target}")"
  case "${target}" in
    interface-lacam3|interface-lagat)
      local preset
      preset="$(preset_for_target "${target}")"
      cmake -S "${source_dir}" --preset "${preset}" -B "${build_dir}"
      (cd "${source_dir}" && cmake --build --preset "${preset}" -- -j"${JOBS}")
      ;;
    planner-lacam3|planner-lagat)
      cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
      cmake --build "${build_dir}" -- -j"${JOBS}"
      ;;
  esac
}

cd "${REPO_ROOT}"

targets=()
case "${TARGET}" in
  all) targets=("interface-lacam3" "interface-lagat" "planner-lacam3" "planner-lagat") ;;
  interfaces) targets=("interface-lacam3" "interface-lagat") ;;
  planners) targets=("planner-lacam3" "planner-lagat") ;;
  interface-lacam3|interface-lagat|planner-lacam3|planner-lagat) targets=("${TARGET}") ;;
esac

if [[ "${DO_DISTCLEAN}" -eq 1 ]]; then
  for target in "${targets[@]}"; do
    run_distclean "${target}"
  done
  exit 0
fi

if [[ "${DO_CLEAN}" -eq 1 ]]; then
  for target in "${targets[@]}"; do
    run_clean "${target}"
  done
  exit 0
fi

for target in "${targets[@]}"; do
  run_build "${target}"
done
