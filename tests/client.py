import socket

HOST = "127.0.0.1"
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print(s.recv(1024).decode(), end="")
    while True:
        msg = input("You: ")
        s.sendall(msg.encode())
        if msg.lower() == "exit":
            break
        data = s.recv(4096).decode()
        print("Bot:", data)
