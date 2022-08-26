## pico.py
# This file wraps up the different picoscope driver python library for easier use of block mode.
# Only supports drivers for ps3000a, ps5000 and ps6000, because I can only test on picoscopes models 6407,5203,3206D.
# WARNING: Is not optimized and might contain bugs.
import numpy as np

from picosdk.errors import *
from picosdk.ps3000a import ps3000a as ps3
from picosdk.ps5000 import ps5000 as ps5
from picosdk.ps5000a import ps5000a as ps5a
from picosdk.ps6000 import ps6000 as ps6
import ctypes
from picosdk.functions import adc2mV, assert_pico_ok

DEBUG_MODE = False

def argClosest(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))

class pico3000():

    def __init__(self):
        print("[*] Picoscope SETUP")
        self.status = {}
        self.chandle = ctypes.c_int16()
        self.chARange = None
        self.voltRange = '',
        self.timeDiv = 0
        self.timebase = 0
        self.voltDiv = 0
        self.sampleRate = 0
        self.maxADC = 0
        self.minADC = 0
        self.channel_out = None
        self.bufferAMin = None
        self.overflow = None
        self.cmaxSamples = None
        self.n_points = 0
    

    def disconnect(self):
        # Stops the scope
        # Handle = chandle
        self.status["stop"] = ps3.ps3000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Closes the unit
        # Handle = chandle
        self.status["close"] = ps3.ps3000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])

    def connect(self):
        self.status["openunit"] = ps3.ps3000aOpenUnit(ctypes.byref(self.chandle), None)
        try:
            assert_pico_ok(self.status["openunit"])
            print("[*] Picoscope CONNECTED")
        except:
            # powerstate becomes the status number of openunit
            powerstate = self.status["openunit"]

            # If powerstate is the same as 282 then it will run this if statement
            if powerstate == 282:
                # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
                self.status["ChangePowerSource"] = ps3.ps3000aChangePowerSource(self.chandle, 282)
                # If the powerstate is the same as 286 then it will run this if statement
            elif powerstate == 286:
                # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
                self.status["ChangePowerSource"] = ps3.ps3000aChangePowerSource(self.chandle, 286)
            else:
                raise

            assert_pico_ok(self.status["ChangePowerSource"])

    def setChannel(self, channel, voltsPerDivision, sampleRate, timeDiv, n_points):
        self.sampleRate = sampleRate
        self.timeDiv = timeDiv
        self.n_points = n_points
        volt_ranges = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        closest_voltDiv = argClosest(volt_ranges, 10*voltsPerDivision*1e3)
        self.voltDiv = (volt_ranges[closest_voltDiv]/4)*1e-3 #update selected voltDiv
        pico_voltDiv = ['50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V']
        self.voltRange = 'PS3000A_'+pico_voltDiv[closest_voltDiv]
        self.chARange = ps3.PS3000A_RANGE['PS3000A_'+pico_voltDiv[closest_voltDiv]]

        self.status["setChA"] = ps3.ps3000aSetChannel(self.chandle, channel, 1, ps3.PS3000A_COUPLING['PS3000A_DC'], self.chARange, 0)# Set up channel A
        assert_pico_ok(self.status["setChA"])
        # Disable other channels
        for ch in range(1,4):
            self.status["setChB"] = ps3.ps3000aSetChannel(self.chandle, (channel + ch)%4, 0, ps3.PS3000A_COUPLING['PS3000A_DC'], self.chARange, 0)
            try:
                assert_pico_ok(self.status["setChB"])
            except(PicoSDKCtypesError):
                # Not all scopes have channels C and D and here will complain.
                pass

        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        # Sample rate for picoscope3000 follows a rule: sampleRate = 1e9/(2**n) for n<3, 125e6/(n-2) for n>=3.
        if self.sampleRate>=250e6:
            self.timebase = max(0, round(np.log2(1e9/self.sampleRate)))# n = log2(1e9/sampleRate) if 1GHz>=sampleRate>=250MHz
            self.sampleRate = 1e9/(2**self.timebase)
        elif sampleRate<250e6:
            self.timebase = min(2**32-1, round(125e6/self.sampleRate+2))# n = 125e6/sampleRate + 2 if sampleRate<125MHz
            self.sampleRate = 125e6/(self.timebase-2)
        else:
            raise 
        self.status["GetTimebase"] = ps3.ps3000aGetTimebase2(self.chandle, self.timebase, self.n_points, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)#get timebase information
        assert_pico_ok(self.status["GetTimebase"])
        self.timeDiv = self.n_points/(10*self.sampleRate)

        # Create buffers ready for assigning pointers for data collection
        self.channel_out = (ctypes.c_int16 * self.n_points)()
        self.bufferAMin = (ctypes.c_int16 * self.n_points)() # used for downsampling which isn't in the scope of this example
        self.status["SetDataBuffers"] = ps3.ps3000aSetDataBuffers(self.chandle, channel, ctypes.byref(self.channel_out), ctypes.byref(self.bufferAMin), self.n_points, 0, 0)# Setting the data buffer location for data collection from channel A
        assert_pico_ok(self.status["SetDataBuffers"])

        # Creates a overlow location for data
        self.overflow = (ctypes.c_int16)()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.n_points)

        # Finds the max ADC count
        self.maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps3.ps3000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
        assert_pico_ok(self.status["maximumValue"])
        self.minADC = ctypes.c_int16()
        self.status["minimumValue"] = ps3.ps3000aMinimumValue(self.chandle, ctypes.byref(self.minADC))
        assert_pico_ok(self.status["minimumValue"])

        if(DEBUG_MODE):
            print("Measure channel")
            print("\tchannel: " + list(ps3.PS3000A_CHANNEL.items())[channel][0])
            print("\tScope state:")
            print("\tsampleRate: {:e}Hz".format(sampleRate))
            print("\tn_points: {}".format(self.n_points))
            print("\tvoltDiv: {}V/div".format(self.voltDiv))
            print("\tvoltRange: "'PS3000A_'+pico_voltDiv[closest_voltDiv])
            print(self.status)
        return self.voltDiv, self.timeDiv, self.sampleRate

    def setTriggerChannel(self, channel, enable=0, threshold=1024, timeout=1000):
        self.status["trigger"] = ps3.ps3000aSetSimpleTrigger(self.chandle, enable, channel, threshold, ps3.PS3000A_THRESHOLD_DIRECTION['PS3000A_RISING'], 0, timeout)# Sets up single trigger
        assert_pico_ok(self.status["trigger"])

        if(DEBUG_MODE):
            print("Trigger channel")
            print("\tchannel: " + list(ps3.PS3000A_CHANNEL.items())[channel][0])
            print("\ttimeout: " + str(timeout))


    def arm(self):
        # Starts block capture
        self.status["runblock"] = ps3.ps3000aRunBlock(self.chandle, 0, self.n_points, self.timebase, 1, None, 0, None, None)
        assert_pico_ok(self.status["runblock"])
        
        if(DEBUG_MODE):
            print("Block capture started.")

    def getNativeSignalBytes(self):
            # Checks data collection to finish the capture
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                self.status["isReady"] = ps3.ps3000aIsReady(self.chandle, ctypes.byref(ready))

            self.status["GetValues"] = ps3.ps3000aGetValues(self.chandle, 0, ctypes.byref(self.cmaxSamples), 0, 0, 0, ctypes.byref(self.overflow))
            assert_pico_ok(self.status["GetValues"])

            # Converts ADC from channel A to mV
            # channel_out_interpreted =  adc2mV(channel_out, chARange, maxADC)

            # Scale the output signal to interpret as bytes
            channel_out_interpreted = np.array([(255 * (x - self.minADC.value) / (self.maxADC.value - self.minADC.value))-(255/2) for x in self.channel_out], dtype='i1')

            return bytes(channel_out_interpreted), channel_out_interpreted

class pico5000():

    def __init__(self):
        print("[*] Picoscope SETUP")
        self.status = {}
        self.chandle = ctypes.c_int16()
        self.chARange = None
        self.timeDiv = 0
        self.timebase = 0
        self.voltDiv = 0
        self.sampleRate = 0
        self.maxADC = 0
        self.minADC = 0
        self.channel_out = None
        self.bufferAMin = None
        self.overflow = None
        self.cmaxSamples = None
        self.n_points = 0
        self.n_points_pre_trigger=0
        self.n_points_post_trigger=0
        self.delay=0
    

    def disconnect(self):
        # Stops the scope
        # Handle = chandle
        self.status["stop"] = ps5.ps5000Stop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Closes the unit
        # Handle = chandle
        self.status["close"] = ps5.ps5000CloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])

    def connect(self):
        self.status["openunit"] = ps5.ps5000OpenUnit(ctypes.byref(self.chandle))
        try:
            assert_pico_ok(self.status["openunit"])
            print("[*] Picoscope CONNECTED")
        except:
            # powerstate becomes the status number of openunit
            powerstate = self.status["openunit"]

            # If powerstate is the same as 282 then it will run this if statement
            if powerstate == 282:
                # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
                self.status["ChangePowerSource"] = ps5.ps5000ChangePowerSource(self.chandle, 282)
                # If the powerstate is the same as 286 then it will run this if statement
            elif powerstate == 286:
                # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
                self.status["ChangePowerSource"] = ps5.ps5000ChangePowerSource(self.chandle, 286)
            else:
                raise

            assert_pico_ok(self.status["ChangePowerSource"])

    def setChannel(self, channel, voltsPerDivision, sampleRate, timeDiv, n_points):
        self.sampleRate = sampleRate
        self.timeDiv = timeDiv
        self.n_points = n_points
        volt_ranges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        closest_voltDiv = argClosest(volt_ranges, 10*voltsPerDivision*1e3)
        self.voltDiv = (volt_ranges[closest_voltDiv]/4)*1e-3 #update selected voltDiv
        pico_voltDiv = ['10MV', '20MV', '50MV', '100MV', '200MV', '500MV', '1V', '2V', '5V', '10V', '20V', '50V']
        self.voltRange = 'PS5000_'+pico_voltDiv[closest_voltDiv]
        self.chARange = ps5.PS5000_RANGE['PS5000_'+pico_voltDiv[closest_voltDiv]]

        self.status["setChA"] = ps5.ps5000SetChannel(self.chandle, channel, 1, True, self.chARange, 0)# Set up channel A, True=DC, False=AC
        assert_pico_ok(self.status["setChA"])
        # Disable other channels
        for ch in range(1,4):
           self.status["setChB"] = ps5.ps5000SetChannel(self.chandle, (channel + ch)%4, 0, True, self.chARange, 0)
        try:
           assert_pico_ok(self.status["setChB"])
        except(PicoSDKCtypesError):
           pass

        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        # Sample rate for picoscope3000 follows a rule: sampleRate = 1e9/(2**n) for n<3, 125e6/(n-2) for n>=3.
        if self.sampleRate>=250e6:
            self.timebase = max(0, round(np.log2(1e9/self.sampleRate)))# n = log2(1e9/sampleRate) if 1GHz>=sampleRate>=250MHz
            self.sampleRate = 1e9/(2**self.timebase)
        elif sampleRate<250e6:
            self.timebase = min(2**32-1, round(125e6/self.sampleRate+2))# n = 125e6/sampleRate + 2 if sampleRate<125MHz
            self.sampleRate = 125e6/(self.timebase-2)
        else:
            raise
        self.status["GetTimebase"] = ps5.ps5000GetTimebase(self.chandle, self.timebase, self.n_points, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)#get timebase information
        assert_pico_ok(self.status["GetTimebase"])
        self.timeDiv = self.n_points/(10*self.sampleRate)

        # Create buffers ready for assigning pointers for data collection
        self.channel_out = (ctypes.c_int16 * self.n_points)()
        self.bufferAMin = (ctypes.c_int16 * self.n_points)() # used for downsampling which isn't in the scope of this example
        self.status["SetDataBuffers"] = ps5.ps5000SetDataBuffers(self.chandle, channel, ctypes.byref(self.channel_out), ctypes.byref(self.bufferAMin), self.n_points, 0, 0) # Setting the data buffer location for data collection from channel A
        assert_pico_ok(self.status["SetDataBuffers"])

        # Creates a overlow location for data
        self.overflow = (ctypes.c_int16)()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.n_points)

        # Finds the max ADC count
        self.maxADC = ctypes.c_int16(32512)
        self.status["maximumValue"] = 0
        assert_pico_ok(self.status["maximumValue"])
        self.minADC = ctypes.c_int16(-32512)
        self.status["minimumValue"] = 0
        assert_pico_ok(self.status["minimumValue"])

        if(DEBUG_MODE):
            print("Measure channel")
            print("\tchannel: " + list(ps5.PS5000_CHANNEL.items())[channel][0])
            print("\tScope state:")
            print("\tsampleRate: {:e}Hz".format(sampleRate))
            print("\tn_points: {}".format(self.n_points))
            print("\tvoltDiv: {}V/div".format(self.voltDiv))
            print("\tvoltRange: "'PS3000A_'+pico_voltDiv[closest_voltDiv])
            print(self.status)
        return self.voltDiv, self.timeDiv, self.sampleRate

    def setTriggerChannel(self, channel, enable=0, threshold=1024, timeout=1000, ratio_points_pre_trigger=0, delay=0):
        self.status["trigger"] = ps5.ps5000SetSimpleTrigger(self.chandle, enable, channel, threshold, 2, ctypes.c_uint32(delay), timeout)# Sets up single trigger
        self.n_points_pre_trigger = int(ratio_points_pre_trigger*self.n_points)
        self.n_points_post_trigger = self.n_points - self.n_points_pre_trigger
        self.delay = delay
        assert_pico_ok(self.status["trigger"])

        if(DEBUG_MODE):
            print("Trigger channel")
            print("\tchannel: " + list(ps5.PS5000_CHANNEL.items())[channel][0])
            print("\ttimeout: " + str(timeout))


    def arm(self):
        MAX_WAVEFORMS = 10
        MAX_SAMPLES = 200000
        ps5a.ps5000aSetNoOfCaptures(self.chandle, MAX_WAVEFORMS)

        # Starts block capture
        self.status["runblock"] = ps5.ps5000RunBlock(self.chandle, self.n_points_pre_trigger, self.n_points_post_trigger, self.timebase, 1, None, 0, None, None)
        assert_pico_ok(self.status["runblock"])

        if(DEBUG_MODE):
            print("Block capture started.")

    def getAllBuffer(self):
            MAX_WAVEFORMS = 10
            MAX_SAMPLES = 200000
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                self.status["isReady"] = ps5.ps5000IsReady(self.chandle, ctypes.byref(ready))

            buffer = [ [ [ctypes.c_int16(0)]*MAX_SAMPLES]*MAX_WAVEFORMS]
            for i in range(MAX_WAVEFORMS):
                ps5a.ps5000aSetDataBuffer(self.chandle, 0, buffer[0][i], MAX_SAMPLES, i, ps5a.PS5000A_RATIO_MODE[0])


    def getNativeSignalBytes(self):
            # Checks data collection to finish the capture
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                self.status["isReady"] = ps5.ps5000IsReady(self.chandle, ctypes.byref(ready))

            self.status["GetValues"] = ps5.ps5000GetValues(self.chandle, 0, ctypes.byref(self.cmaxSamples), 0, 0, 0, ctypes.byref(self.overflow))
            assert_pico_ok(self.status["GetValues"])

            #Converts ADC from channel A to mV
            #channel_out_interpreted =  adc2mV(channel_out, chARange, maxADC)

            #Scale the output signal to interpret as bytes
            channel_out_interpreted = np.array([(255 * (x - self.minADC.value) / (self.maxADC.value - self.minADC.value))-(255/2) for x in self.channel_out], dtype='i1')

            return bytes(channel_out_interpreted), channel_out_interpreted

class pico6000():

    def __init__(self):
        print("[*] Picoscope SETUP")
        self.status = {}
        self.chandle = ctypes.c_int16()
        self.chARange = None
        self.timeDiv = 0
        self.timebase = 0
        self.voltDiv = 0
        self.sampleRate = 0
        self.maxADC = 0
        self.minADC = 0
        self.channel_out = None
        self.bufferAMin = None
        self.overflow = None
        self.cmaxSamples = None
        self.n_points = 0
    

    def disconnect(self):
        # Stops the scope
        # Handle = chandle
        self.status["stop"] = ps6.ps6000Stop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Closes the unit
        # Handle = chandle
        self.status["close"] = ps6.ps6000CloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])

    def connect(self):
        self.status["openunit"] = ps6.ps6000OpenUnit(ctypes.byref(self.chandle), None, 0)#resolution, 0=8bits, 
        try:
            assert_pico_ok(self.status["openunit"])
            print("[*] Picoscope CONNECTED")
        except:
            # powerstate becomes the status number of openunit
            powerstate = self.status["openunit"]

            # If powerstate is the same as 282 then it will run this if statement
            if powerstate == 282:
                # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
                self.status["ChangePowerSource"] = ps6.ps6000ChangePowerSource(self.chandle, 282)
                # If the powerstate is the same as 286 then it will run this if statement
            elif powerstate == 286:
                # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
                self.status["ChangePowerSource"] = ps6.ps6000ChangePowerSource(self.chandle, 286)
            else:
                raise

            assert_pico_ok(self.status["ChangePowerSource"])

    def setChannel(self, channel, voltsPerDivision, sampleRate, timeDiv, n_points):
        self.sampleRate = sampleRate
        self.timeDiv = timeDiv
        self.n_points = n_points
        # volt_ranges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        # closest_voltDiv = argClosest(volt_ranges, 10*voltsPerDivision*1e3)
        # self.voltDiv = (volt_ranges[closest_voltDiv]/8)*1e-3 #update selected voltDiv
        # pico_voltDiv = ['A_10MV', '_20MV', '_50MV', '_100MV', '_200MV', '_500MV', '_1V', '_2V', '_5V', '_10V', '_20V', '_50V']

        #Picoscope 6407 only have +-100mv range
        self.voltDiv = (1000/4)*1e-3 #update selected voltDiv
        self.chARange = ps6.PS6000_RANGE['PS6000_100MV']
        self.voltRange = 'PS6000_100MV'
        
        self.status["setChA"] = ps6.ps6000SetChannel(self.chandle, channel, 1, ps6.PS6000_COUPLING['PS6000_DC_50R'], self.chARange, 0, ps6.PS6000_BANDWIDTH_LIMITER["PS6000_BW_FULL"])# Set up channel A
        assert_pico_ok(self.status["setChA"])
        # Disable other channels
        for ch in range(1,4):
            self.status["setChB"] = ps6.ps6000SetChannel(self.chandle, (channel + ch)%4, 0, ps6.PS6000_COUPLING['PS6000_DC_50R'], self.chARange, 0, ps6.PS6000_BANDWIDTH_LIMITER["PS6000_BW_FULL"])
        try:
            assert_pico_ok(self.status["setChB"])
        except(PicoSDKCtypesError):
            pass

        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        # Sample rate for picoscope6000 follows a rule: sampleRate = 5e9/(2**n) for n<5, 156.250e6/(n-4) for n>=5.
        if self.sampleRate>=156.25e6:
            self.timebase = max(0, round(np.log2(5e9/self.sampleRate)))# n = log2(5e9/sampleRate) if 5GHz>=sampleRate>=156.25MHz
            self.sampleRate = 5e9/(2**self.timebase)
        elif sampleRate<156.25e6:
            self.timebase = min(2**32-1, round(156.25e6/self.sampleRate+4))# n = 156.25e6/sampleRate + 4 if sampleRate<125MHz
            self.sampleRate = 156.25e6/(self.timebase-4)
        else:
            raise 
        self.status["GetTimebase"] = ps6.ps6000GetTimebase2(self.chandle, self.timebase, self.n_points, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)#get timebase information
        assert_pico_ok(self.status["GetTimebase"])
        self.timeDiv = self.n_points/(10*self.sampleRate)

        # Create buffers ready for assigning pointers for data collection
        self.channel_out = (ctypes.c_int16 * self.n_points)()
        self.bufferAMin = (ctypes.c_int16 * self.n_points)() # used for downsampling which isn't in the scope of this example
        self.status["SetDataBuffers"] = ps6.ps6000SetDataBuffers(self.chandle, channel, ctypes.byref(self.channel_out), ctypes.byref(self.bufferAMin), self.n_points, 0, 0)# Setting the data buffer location for data collection from channel A
        assert_pico_ok(self.status["SetDataBuffers"])

        # Creates a overlow location for data
        self.overflow = (ctypes.c_int16)()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.n_points)

        # Finds the max ADC count
        self.maxADC = ctypes.c_int16(32512)
        self.status["maximumValue"] = 0
        assert_pico_ok(self.status["maximumValue"])
        self.minADC = ctypes.c_int16(-32512)
        self.status["minimumValue"] = 0
        assert_pico_ok(self.status["minimumValue"])

        if(DEBUG_MODE):
            print("Measure channel")
            print("\tchannel: " + list(ps6.PS6000_CHANNEL.items())[channel][0])
            print("\tScope state:")
            print("\tsampleRate: {:e}Hz".format(sampleRate))
            print("\tn_points: {}".format(self.n_points))
            print("\tvoltDiv: {}V/div".format(self.voltDiv))
            print("\tvoltRange: "'PS6000A_'+pico_voltDiv[closest_voltDiv])
            print(self.status)
        return self.voltDiv, self.timeDiv, self.sampleRate

    def setTriggerChannel(self, channel, enable=0, threshold=1024, timeout=1000):
        self.status["trigger"] = ps6.ps6000SetSimpleTrigger(self.chandle, enable, channel, threshold, ps6.PS6000_THRESHOLD_DIRECTION['PS6000_RISING'], 0, timeout)# Sets up single trigger
        assert_pico_ok(self.status["trigger"])

        if(DEBUG_MODE):
            print("Trigger channel")
            print("\tchannel: " + list(ps6.PS6000_CHANNEL.items())[channel][0])
            print("\ttimeout: " + str(timeout))


    def arm(self):
        # Starts block capture
        self.status["runblock"] = ps6.ps6000RunBlock(self.chandle, 0, self.n_points, self.timebase, 1, None, 0, None, None)
        assert_pico_ok(self.status["runblock"])
        
        if(DEBUG_MODE):
            print("Block capture started.")

    def getNativeSignalBytes(self):
            # Checks data collection to finish the capture
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                self.status["isReady"] = ps6.ps6000IsReady(self.chandle, ctypes.byref(ready))

            self.status["GetValues"] = ps6.ps6000GetValues(self.chandle, 0, ctypes.byref(self.cmaxSamples), 1, 0, 0, ctypes.byref(self.overflow))
            
            assert_pico_ok(self.status["GetValues"])

            # Converts ADC from channel A to mV
            # channel_out_interpreted =  adc2mV(channel_out, chARange, maxADC)

            # Scale the output signal to interpret as bytes
            channel_out_interpreted = np.array([(255 * (x - self.minADC.value) / (self.maxADC.value - self.minADC.value))-(255/2) for x in self.channel_out], dtype='i1')

            return bytes(channel_out_interpreted), channel_out_interpreted
