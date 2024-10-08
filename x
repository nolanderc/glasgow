#!/usr/bin/env bash

set -eu -o pipefail

function list {
    echo "commands:"
    declare -F | awk '{ print "  " $3 }'
}

function build {
    cargo build "$@"
}

function test {
    cargo nextest run "$@"
}

function test-snapshot {
    export UPDATE_EXPECT=1
    cargo nextest run "$@"
}

function run {
    cargo run "$@"
}

function check {
    cargo clippy "$@"
}

function install {
    cargo install --path .
}

function watch {
    watchexec -e rs,toml -c --on-busy-update=restart -- ./x "$@"
}

function exec {
    "$@"
}

if [[ $# == 0 ]]; then
    list
    exit 1
elif declare -f "$1" > /dev/null; then
    "$@"
    exit $?
else
    echo "'$1' is not a known command"
    list
    exit 1
fi
