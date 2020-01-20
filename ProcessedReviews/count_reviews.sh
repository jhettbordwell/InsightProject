#!/bin/bash

FILES="*reviews.csv"
TOTAL=0
for f in $FILES
do
    COUNT=`wc -l < $f`
    echo $COUNT
    TOTAL=$((TOTAL + COUNT))
done

echo $TOTAL
