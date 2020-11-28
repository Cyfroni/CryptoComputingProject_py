import random
import selectors
import socket
import sys
import types

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432  # The port used by the server


class Client:
    def __init__(self, host, port):
        self.serverHost = host
        self.serverPort = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sel = selectors.DefaultSelector()
        self.connId = random.randint(1, 1048576)

    def start_connections(self, messages):
        server_addr = (self.serverHost, self.serverPort)
        print('starting connection to', server_addr)
        self.socket.setblocking(False)
        self.socket.connect_ex(server_addr)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(connid=self.connId,
                                     msg_total=sum(len(m) for m in messages),
                                     recv_total=0,
                                     messages=list(messages),
                                     outb=b'')
        self.sel.register(self.socket, events, data=data)

    def service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                print('received', repr(recv_data), 'from connection', data.connid)
                data.recv_total += len(recv_data)
            if not recv_data or data.recv_total == data.msg_total:
                print('closing connection', data.connid)
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if not data.outb and data.messages:
                data.outb = data.messages.pop(0)
            if data.outb:
                print("sending", repr(data.outb), "to connection", data.connid)
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]

"""
if len(sys.argv) != 4:
    print("usage:", sys.argv[0], "<host> <port> <message>")
    sys.exit(1)

host, port, message = sys.argv[1:4]
messages = [str(message).encode()]  # Byte stream need to contains the secret shares, MACs and other stuff
messages = [b'shares 12121212 68556112332 98152002556']
macs = [b'macs 7876543524 12645647983 35648978125']
client = Client(host, int(port))
client.start_connections(messages)

# start_connections(host, int(port), int(num_conns);
try:
    while True:
        events = client.sel.select(timeout=1)
        if events:
            for key, mask in events:
                client.service_connection(key, mask)
        # Check for a socket being monitored to continue.
        if not client.sel.get_map():
            break
except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    client.sel.close()
"""