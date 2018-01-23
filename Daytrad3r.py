#!/usr/bin/python
from pymongo import MongoClient
from time import sleep
import gdax


class WebSocketClient(gdax.WebsocketClient):
    def on_open(self):
        self.url = "wss://ws-feed.gdax.com/"
        self.products = ["BTC-USD"]
        self.message_count = 0

    def on_message(self, msg):
        self.message_count += 1
        print(msg)
    def on_close(self):
        print("-- Goodbye! --")


def get_accounts(client):
    l = client.get_accounts()
    print("Account      Balance      ")
    print("---------------------------")
    for account in l:
        print(account['currency'] + "          " + account['balance'])

def btc_to_usd(btc):
    ws_client = WebSocketClient()
    ws_client.start()
    time.sleep(.5)
    ws_client.close()




def sell(client, price, size, prod_id):
    client.sell(price=price, size=size, product_id=prod_id)

def main():
    stuff = []

    with open(".idek") as f:
        for line in f.readlines():
            stuff.append(line.rstrip())

    auth_client = gdax.AuthenticatedClient(stuff[1], stuff[2], stuff[0], api_url="https://api-public.sandbox.gdax.com")

    #sell(auth_client, '4000', '0.01', 'BTC-USD')

    get_accounts(auth_client)

if __name__ == '__main__':
    main()
