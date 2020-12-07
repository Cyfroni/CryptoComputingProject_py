import random
import selectors
import socket
import sys
import types
from threading import Thread

from . import client

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


def print(*args, **kwargs): pass


class Party:
    def connect(self, serv, partyId, otherServers, message):
        self.serv = serv
        self.client = None
        self.partyId = partyId
        self.otherServers = otherServers
        self.message = message

    def start_client(self, server_index):
        messages = [b"shares 1 12121212 68556112332 98152002556"]
        macs = [b"macs 1 7876543524 12645647983 35648978125"]
        self.client = client.Client(self.otherServers[server_index].HOST, int(
            self.otherServers[server_index].PORT))
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
        # filtered_parties = [
        #     temp_party for temp_party in parties if temp_party.partyId == partyId]
        # destination_party = filtered_parties[0]
        self.client = client.Client(HOST, int(PORT + partyId))
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

# if __name__ == "__main__":
#     numberOfParties = random.randint(2, 5)
#     parties = parties_init(numberOfParties)

#     for i in range(0, numberOfParties):
#         parties[i].start_client(random.randint(0, numberOfParties - 2))
