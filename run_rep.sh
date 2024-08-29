#!/bin/bash

run_script() {
    SCRIPT_ARGS=$1
    while true; do
        echo "Starting $SCRIPT_ARGS..."
        python $SCRIPT_ARGS
        
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "Script $SCRIPT_ARGS ended normally."
            return 0
        else
            echo "Script $SCRIPT_ARGS crashed with exit code $STATUS. Restarting in 10 seconds..."
            sleep 10
        fi
    done
}

if [ -z "$1" ]; then
    echo "No script specified. Exiting..."
    exit 1
fi

run_script "$1"

if [ -n "$2" ]; then
    run_script "$2"
fi

if [ -n "$3" ]; then
    run_script "$3"
fi

if [ -n "$4" ]; then
    run_script "$4"
fi

if [ -n "$5" ]; then
    run_script "$5"
fi

if [ -n "$6" ]; then
    run_script "$6"
fi

echo "All specified scripts have ended normally. Exiting..."
exit 0