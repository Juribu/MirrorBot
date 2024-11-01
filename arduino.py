import serial

arduino = serial.Serial(port='/dev/cu.usbmodem1401', baudrate=9600)

while True:
    action = input("Enter input: ")
    arduino.write(bytes(action, 'utf-8'))
