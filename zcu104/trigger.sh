#!/bin/bash

echo 1 > /sys/class/gpio/gpio511/value
sleep 0.01
echo 0 > /sys/class/gpio/gpio511/value