#!/bin/sh

cp -r log_$1 log_$2
cd log_$2
cp infos_$1-best.pkl infos_$2-best.pkl 
cp infos_$1.pkl infos_$2.pkl 
cd ../