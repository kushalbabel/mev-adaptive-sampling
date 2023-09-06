import selectors
import socket

servers = [
            ('localhost', 1235),
            ('localhost', 1236),
            ('localhost', 1237),
            ('localhost', 1238),
            ('localhost', 1239),
        ]

sel = selectors.DefaultSelector()

def read(conn, mask):
    data = conn.recv(4096)  # Should be ready
    if data:
        print('echoing', repr(data), 'to', conn)
    else:
        print('closing', conn)
        sel.unregister(conn)
        conn.close()

sockets = []
for i in range(len(servers)):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((servers[i][0], servers[i][1]))
    sockets.append(s)
    sel.register(s, selectors.EVENT_READ, read)

for num in range(1000000):
    for i in range(len(servers)):
        sockets[i].sendall(bytes("Hello, server " + str(i) + " in round " + str(num), 'utf-8'))
        
    numrep = 0
    while numrep < len(servers):
        events = sel.select(None)
        for key, mask in events:
            numrep += 1
            callback = key.data
            callback(key.fileobj, mask)



