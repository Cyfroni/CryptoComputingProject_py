import random
import selectors
import socket
import sys
import types
from threading import Thread

import server
import client


class Party:
    def __init__(self, serv, partyId, otherServers, message):
        self.serv = serv
        self.client = None
        self.partyId = partyId
        self.otherServers = otherServers
        self.message = message
        self.received_data = {}

    def start_client(self):
        messages = [b"shares 1 12121212 68556112332 98152002556"]
        macs = [b"macs 1 7876543524 12645647983 35648978125"]
        server_index = random.randint(0, numberOfParties - 2)
        self.client = client.Client(self.otherServers[server_index].HOST, int(self.otherServers[server_index].PORT))
        print("From: ", (self.serv.HOST, self.serv.PORT), "\tTo: ",
              (self.otherServers[server_index].HOST, int(self.otherServers[server_index].PORT)))
        self.client.start_connections(messages)
        self.handle_client()

    def broadcast_message(self, message):
        for i in self.otherServers:
            self.client = client.Client(i.HOST, int(i.PORT))
            self.client.start_connections(message)
            self.handle_client()

    def unicast_message(self, partyId, message):
        filtered_parties = [temp_party for temp_party in parties if temp_party.partyId == partyId]
        destination_party = filtered_parties[0]
        self.client = client.Client(destination_party.serv.HOST, int(destination_party.serv.PORT))
        self.client.start_connections(message)
        self.handle_client()

    def handle_client(self):
        try:
            while True:
                events = self.client.sel.select(timeout=1)
                if events:
                    for key, mask in events:
                        self.client.service_connection(key, mask)
                # Check for a socket being monitored to continue.
                if not self.client.sel.get_map():
                    break
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
        finally:
            self.client.sel.close()


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

numberOfParties = random.randint(2, 5)
servers = []
for i in range(0, numberOfParties):
    serv = server.Server(HOST, PORT + i)
    servers.append(serv)
    serv.start()

parties = []
for i in range(0, numberOfParties):
    otherServers = servers[:]
    otherServers.remove(servers[i])
    party = Party(servers[i], i, otherServers, None)
    print("Server: ", (party.serv.HOST, party.serv.PORT), "\tIndex: ", party.partyId)
    [print("\tOther servers:", (i.HOST, i.PORT)) for i in otherServers]
    parties.append(party)

for i in range(0, numberOfParties):
    parties[i].start_client()
