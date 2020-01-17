#!/bin/bash
for file in *.csv; do
    cat $file | kafkacat -P -b localhost -t ${file%.*}
    echo "Published $file to ${file%.*}"
done