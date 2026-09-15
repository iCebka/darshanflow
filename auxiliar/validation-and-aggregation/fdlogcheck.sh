#!/bin/bash

for LOGFILE in "$@"; do

    TXTFILE="${LOGFILE}.txt"

    darshan-parser --show-incomplete "$LOGFILE" > "$TXTFILE"

    if grep -Fq "module contains incomplete data" "$TXTFILE"; then
        echo "INCOMPLETE - $LOGFILE"
    else
        echo "OK - $LOGFILE"
    fi

done
