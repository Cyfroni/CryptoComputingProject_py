import selectors
import socket
import sys
import threading
import types


def print(*args, **kwargs): pass


class Server(threading.Thread):
    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sel = selectors.DefaultSelector()
        self.received_data = {}
        threading.Thread.__init__(self)

    def run(self):
        server = Server(self.HOST, self.PORT)
        server.socket.bind((self.HOST, self.PORT))
        server.socket.listen()
        print("listening on", (self.HOST, self.PORT))
        server.socket.setblocking(False)
        server.sel.register(server.socket, selectors.EVENT_READ, data=None)

        try:
            while True:
                events = server.sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        server.accept_wrapper()
                    else:
                        result = server.service_connection(key, mask)
                        if result is not None:
                            result_str = str(result, encoding='utf-8')
                            elements = result_str.split()
                            type = elements[0]
                            partyId = elements[1]
                            remaining = elements[2:]
                            self.received_data[partyId] = remaining
                            print("Server service connection result: ",
                                  result_str, "\tType: ", type, "\tpartyId: ", partyId)
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
        finally:
            server.sel.close()

    def accept_wrapper(self):
        conn, addr = self.socket.accept()  # Should be ready to read
        print('accepted connection from', addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, events, data=data)

    def service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                data.outb += recv_data
                print([recv_data])
                return data.outb
            else:
                print('closing connection to', data.addr)
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if data.outb:
                print('echoing', repr(data.outb), 'to', data.addr)
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]


"""
if len(sys.argv) != 3:
    print("usage:", sys.argv[0], "<host> <port>")
    sys.exit(1)

host, port = sys.argv[1], int(sys.argv[2])
server = Server(host, port)
server.socket.bind((host, port))
server.socket.listen()
print("listening on", (host, port))
server.socket.setblocking(False)
server.sel.register(server.socket, selectors.EVENT_READ, data=None)

try:
    while True:
        events = server.sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                server.accept_wrapper()
            else:
                server.service_connection(key, mask)
except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    server.sel.close()
"""
