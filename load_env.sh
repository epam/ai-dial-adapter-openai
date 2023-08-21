#!/bin/sh

function load_env() {
  set -o allexport
  source ./.env
  set +o allexport
}