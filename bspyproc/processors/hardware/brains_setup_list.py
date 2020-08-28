class BrainsDevices():
    
    def setup_devices_brains():
    
        devices_dict = {}
        
        devices_dict["A"] = {}
        devices_dict["A"]["input_channel"] = "cDAQ1Mod4/ai0"
        devices_dict["A"]["output_channels"] = [ "cDAQ1Mod2/ao6", "cDAQ1Mod2/ao0",
                                                 "cDAQ1Mod2/ao7", "cDAQ1Mod2/ao5",
                                                 "cDAQ1Mod2/ao2", "cDAQ1Mod2/ao4",
                                                 "cDAQ1Mod2/ao3" ]
        devices_dict["B"] = {}
        devices_dict["B"]["input_channel"] = "cDAQ1Mod4/ai3"
        devices_dict["B"]["output_channels"] = [ "cDAQ1Mod1/ao4", "cDAQ1Mod1/ao3",
                                                 "cDAQ1Mod1/ao5", "cDAQ1Mod1/ao2",
                                                 "cDAQ1Mod1/ao0", "cDAQ1Mod1/ao7",
                                                 "cDAQ1Mod1/ao1" ]
        devices_dict["C"] = {}
        devices_dict["C"]["input_channel"] = "cDAQ1Mod4/ai2"
        devices_dict["C"]["output_channels"] = [ "cDAQ1Mod3/ao14", "cDAQ1Mod3/ao7",
                                                 "cDAQ1Mod3/ao13", "cDAQ1Mod3/ao8",
                                                 "cDAQ1Mod3/ao10", "cDAQ1Mod3/ao11",
                                                 "cDAQ1Mod3/ao12" ]
        devices_dict["D"] = {}
        devices_dict["D"]["input_channel"] = "cDAQ1Mod4/ai4"
        devices_dict["D"]["output_channels"] = [ "cDAQ1Mod3/ao0", "cDAQ1Mod3/ao2",
                                                 "cDAQ1Mod3/ao5", "cDAQ1Mod3/ao3",
                                                 "cDAQ1Mod3/ao4", "cDAQ1Mod3/ao6",
                                                 "cDAQ1Mod3/ao1" ]
        devices_dict["E"] = {}
        devices_dict["E"]["input_channel"] = "cDAQ1Mod4/ai1"
        devices_dict["E"]["output_channels"] = [ "cDAQ1Mod1/ao10", "cDAQ1Mod1/ao13",
                                                 "cDAQ1Mod1/ao9" , "cDAQ1Mod1/ao14",
                                                 "cDAQ1Mod1/ao8" , "cDAQ1Mod1/ao15",
                                                 "cDAQ1Mod1/ao11" ]
        return devices_dict