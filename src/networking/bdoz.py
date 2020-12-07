
from . import server
from .party import Party

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


def print(*args, **kwargs): pass


class BDOZParty(Party):

    def _get_messages(self):
        messages = self.serv.received_data
        self.serv.received_data = {}
        return messages

    def _get_message(self, source_id):
        message = self.serv.received_data[str(source_id)]
        self.serv.received_data[str(source_id)] = None
        return message

    def _to_message(self, vals):
        message = " ".join(map(str, vals))
        return [bytes(f"0 {self.partyId} {message}", 'ascii')]

    def _to_vals(self, message):
        return list(map(int, message))

    def _broadcast(self, vals):
        print(f"{self.partyId}: broadcast {vals}")
        self.broadcast_message(self._to_message(vals))

    def _receive_broadcast(self, vals):
        messages = self._get_messages()
        print(f"{self.partyId}: received {messages}")

        ret = [0] * (len(messages) + 1)

        for party_id, message in messages.items():
            ret[int(party_id)] = self._to_vals(message)

        ret[self.partyId] = vals
        return ret

    def _unicast(self, target_id, vals):
        print(f"{self.partyId} -> {target_id}: unicast {vals}")
        self.unicast_message(target_id, self._to_message(vals))

    def _receive_unicast(self, source_id):
        message = self._get_message(source_id)
        print(f"{self.partyId}: received {message}")

        return self._to_vals(message)


def parties_connect(parties):
    servers = []
    for i in range(len(parties)):
        serv = server.Server(HOST, PORT + i)
        servers.append(serv)
        serv.start()

    for i, party in enumerate(parties):
        otherServers = servers[:]
        otherServers.remove(servers[i])
        party.connect(servers[i], i, otherServers, None)
        print("Server: ", (party.serv.HOST, party.serv.PORT),
              "\tIndex: ", party.partyId)
        [print("\tOther servers:", (s.HOST, s.PORT)) for s in otherServers]
