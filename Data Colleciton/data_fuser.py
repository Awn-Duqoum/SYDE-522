# usage: python data_fuser.py [mac1] [mac2] ... [mac(n)]
from __future__ import print_function
from ctypes import c_void_p, cast, POINTER
from mbientlab.metawear import MetaWear, libmetawear, parse_value, cbindings
from time import sleep
from threading import Event
from sys import argv

import platform
import sys
import keyboard 
import time
import os

dataFolder = "data/"+str(int(time.time()))
os.mkdir(dataFolder)

states = []

class State:
    def __init__(self, device):
        self.device = device
        self.callback = cbindings.FnVoid_VoidP_DataP(self.data_handler)
        self.processor = None
        # 0 - Wait to start collecting data
        # 1 - Collect data
        # 2 - End program
        self.myStatus = 0
        self.fileIndex = 0
        self.outputFile = 0
        self.dataFolder = dataFolder
        
    def openFile(self):
        self.outputFile = open(dataFolder + "/" + str(int(time.time())) + ".json", "w")
        self.outputFile.write('{ "data":[')
        
    def closeFile(self):
        self.outputFile.write('{{"x":{},"y":{},"z":{}}}'.format(-100, -100, -100))
        self.outputFile.write(']}')
        self.outputFile.close()
        
    def data_handler(self, ctx, data):
        values = parse_value(data, n_elem = 2)
        if(self.myStatus == 1):
            self.outputFile.write('{{ "Accel":{{"x":{},"y":{},"z":{}}}, "Gyro":{{"x":{},"y":{},"z":{}}} }},'.format(values[0].x, values[0].y, values[0].z, values[1].x, values[1].y, values[1].z))
        else:
            print("acc: (%.4f,%.4f,%.4f), gyro; (%.4f,%.4f,%.4f)" % (values[0].x, values[0].y, values[0].z, values[1].x, values[1].y, values[1].z))

    def setup(self):
        libmetawear.mbl_mw_settings_set_connection_parameters(self.device.board, 7.5, 7.5, 0, 6000)
        sleep(1.5)

        e = Event()

        def processor_created(context, pointer):
            self.processor = pointer
            e.set()
        fn_wrapper = cbindings.FnVoid_VoidP_VoidP(processor_created)
        
        # Steam the accelerometer at 100 HZ
        libmetawear.mbl_mw_acc_set_odr(s.device.board, 100.0)
        # Set the range to 16 G
        libmetawear.mbl_mw_acc_set_range(s.device.board, 16.0)
        # Write the config to the board
        libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board)
        
        acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
        gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)

        signals = (c_void_p * 1)()
        signals[0] = gyro
        libmetawear.mbl_mw_dataprocessor_fuser_create(acc, signals, 1, None, fn_wrapper)
        e.wait()

        libmetawear.mbl_mw_datasignal_subscribe(self.processor, None, self.callback)

    def start(self):
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(self.device.board)
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.device.board)

        libmetawear.mbl_mw_gyro_bmi160_start(self.device.board)
        libmetawear.mbl_mw_acc_start(self.device.board)
        
for i in range(len(argv) - 1):
    d = MetaWear(argv[i + 1])
    d.connect()
    print("Connected to " + d.address)
    states.append(State(d))

for s in states:
    print("Configuring %s" % (s.device.address))
    s.setup()

for s in states:
    s.start()

while(states[0].myStatus != 2):
    keyboard.wait('space')
    states[0].openFile()
    states[0].myStatus = 1
    print("********\nCollecting Data")
    keyboard.wait('space')
    states[0].closeFile() 
    states[0].myStatus = 0
    print("Closing Folder")
    print("Going to bed - Hold Escape to End")
    sleep(0.500)
    print("Waking Up - Escape will do nothing now")
    if(keyboard.is_pressed('esc')):
        states[0].myStatus = 2
        print("The escape button is pressed")
           
print("Resetting devices")
events = []
for s in states:
    e = Event()
    events.append(e)

    s.device.on_disconnect = lambda s: e.set()
    libmetawear.mbl_mw_debug_reset(s.device.board)

for e in events:
    e.wait()