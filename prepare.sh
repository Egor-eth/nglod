#!/bin/bash

VENV_DIR=.venv


#Virtual env doesnt exist -> install deps
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR" || rm -rf "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi


#Virtual env exists -> load
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

pip install -r infra/requirements.txt &&
touch "$VENV_DIR/installed" && {
    cd sdf-net/lib/extensions &&
    chmod +x build_ext.sh && ./build_ext.sh
    cd ../../..
}