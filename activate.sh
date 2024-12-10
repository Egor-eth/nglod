#!/bin/bash

VENV_DIR=.venv

#Virtual env doesnt exist -> run prepare
if [ ! -f "$VENV_DIR/installed" ]; then
	/bin/bash ./prepare.sh
fi

#Virtual env exists -> load
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi
