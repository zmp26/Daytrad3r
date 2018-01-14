#!/usr/bin/python
import gdax

stuff = []

with open(".idek") as f:
    for line in f.readlines():
        stuff.append(line.rstrip())

auth_client = gdax.AuthenticatedClient(stuff[1], stuff[2], stuff[0])

print(auth_client.get_accounts())
