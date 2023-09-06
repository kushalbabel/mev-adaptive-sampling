PYTHON=python

$PYTHON server.py &
$PYTHON server.py 1236 &
$PYTHON server.py 1237 &
$PYTHON server.py 1238 &
$PYTHON server.py 1239 &
sleep 2
$PYTHON coordinator.py
sleep 2
