import socket
import sys
import time
import random

HOST = "127.0.0.1"
PORT = 1235
if len(sys.argv) == 2:
    PORT = int(sys.argv[1])


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    s.setblocking(True)
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(4096)
            if not data:
                break
            print(data)
            time.sleep(random.randrange(1, 3))
            conn.sendall(data)

