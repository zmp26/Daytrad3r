#!/usr/bin/python
import gdax

stuff = []

with open("~/Documents/.idek") as f:
    for line in f.readLines():
        stuff.append(line.rstrip())

print stuff
