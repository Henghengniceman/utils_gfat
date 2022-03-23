import milos

class CiaoMixin:

    def get_PT(self):
        ''' Gets the pressure and temperature at station level from the Milos station.
        The results are stored in the info dictionary.        
        '''
        
        start_time = self.info['start_time']
        stop_time = self.info['stop_time']
        dt = stop_time - start_time
        mean_time = start_time + dt/2
        
        # this guarantees that more that half the measurement period is taken into account
        atm = milos.Atmospheric_condition(mean_time)
        temperature = atm.get_mean('Air_Temperature', start_time, stop_time)
        pressure = atm.get_mean('Air_Pressure', start_time, stop_time)
        self.info['Temperature'] = temperature
        self.info['Pressure'] = pressure
